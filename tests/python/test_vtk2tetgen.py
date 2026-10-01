"""Unit tests for VTK2tetgen's fiber/sheet orthonormalization math.

createONS and NormFiberSheetNormal are pure (numpy only); they build the local
material basis written to the .bases file. A wrong basis silently corrupts the
mechanics, so orthonormality and right-handedness are worth pinning.
"""
import meshio
import numpy as np
import pytest

import VTK2tetgen as vt


def _unpack(m):
    m = np.asarray(m, dtype=float)
    return m[0:3], m[3:6], m[6:9]


FIBERS = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],   # axis-aligned: exercises each seed branch
    [1, 1, 0], [2, 3, 6], [-1, 2, -2], [0.1, 0.0, 0.0],
]


@pytest.mark.parametrize("f", FIBERS)
def test_createons_orthonormal_righthanded(f):
    fn, s, sn = _unpack(vt.createONS(f))
    # unit length
    for v in (fn, s, sn):
        assert np.isclose(np.linalg.norm(v), 1.0)
    # mutually orthogonal
    assert np.isclose(np.dot(fn, s), 0.0, atol=1e-12)
    assert np.isclose(np.dot(fn, sn), 0.0, atol=1e-12)
    assert np.isclose(np.dot(s, sn), 0.0, atol=1e-12)
    # fiber direction preserved (just normalized)
    assert np.allclose(fn, np.asarray(f, float) / np.linalg.norm(f))
    # right-handed: f x s == sn and det == +1
    assert np.allclose(np.cross(fn, s), sn, atol=1e-12)
    assert np.isclose(np.linalg.det(np.array([fn, s, sn])), 1.0)


def test_createons_matches_identity_for_x_axis():
    assert np.allclose(vt.createONS([1, 0, 0]), [1, 0, 0, 0, 1, 0, 0, 0, 1])


def test_normfibersheetnormal_normalizes_and_preserves_direction():
    f, s, sn = [2, 0, 0], [0, 3, 0], [0, 0, 4]
    fn, sfn, snn = _unpack(vt.NormFiberSheetNormal(f, s, sn))
    assert np.allclose(fn, [1, 0, 0])
    assert np.allclose(sfn, [0, 1, 0])
    assert np.allclose(snn, [0, 0, 1])


def test_normfibersheetnormal_keeps_nonorthogonal_input():
    # It only normalizes; a non-orthogonal input stays non-orthogonal (no re-basing).
    fn, s, sn = _unpack(vt.NormFiberSheetNormal([1, 1, 0], [0, 1, 0], [0, 0, 1]))
    assert np.isclose(np.linalg.norm(fn), 1.0)
    assert not np.isclose(np.dot(fn, s), 0.0)


# --- Conversion of tiny VTUs to files ---------------------------------------

UNIT_TET = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
# VTK quadratic-tet edge order: (0,1) (1,2) (0,2) (0,3) (1,3) (2,3)
T10_POINTS = np.vstack([UNIT_TET, [(UNIT_TET[a] + UNIT_TET[b]) / 2
                                   for a, b in [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]]])
PLAIN_FULL = {"Fiber": [1, 2, 2], "Sheet": [0, 3, 0], "Sheetnormal": [0, 0, 4]}
# Golden basis lines that conversion from plain arrays must keep reproducing byte for byte.
BASIS_FROM_FIBER = ("0.3333333333333333 0.6666666666666666 0.6666666666666666 0.9428090415820634 "
                    "-0.23570226039551584 -0.23570226039551584 0.0 0.7071067811865475 -0.7071067811865475")
BASIS_FROM_FULL = "0.3333333333333333 0.6666666666666666 0.6666666666666666 0.0 1.0 0.0 0.0 0.0 1.0"
# Each point's basis is a distinct permutation of scaled axes, so a swapped
# point order or component shows up in the output.
AXIS_PERMUTATIONS = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0)]


def _per_point_arrays():
    arrays = {}
    for q, perm in enumerate(AXIS_PERMUTATIONS):
        for c, axis in zip(vt.BASIS_COMPONENTS, perm):
            arrays[f"{c}_{q}"] = list((q + 2) * np.eye(3)[axis])
    return arrays


def _convert(tmp_path, monkeypatch, t10, arrays, *extra_args):
    """Write a one-element VTU carrying `arrays` as cell data, convert it, return the output prefix."""
    points, cell_type = (T10_POINTS, "tetra10") if t10 else (UNIT_TET, "tetra")
    cell_data = {name: [np.array([value], float)] for name, value in arrays.items()}
    cell_data["Material"] = [np.array([7])]
    vtu = tmp_path / "mesh.vtu"
    meshio.write(vtu, meshio.Mesh(points, [(cell_type, [list(range(len(points)))])], cell_data=cell_data))
    prefix = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["VTK2tetgen.py", str(vtu), "-outfile", str(prefix), *extra_args])
    vt.main()
    return prefix


@pytest.mark.parametrize("t10", [False, True], ids=["T4", "T10"])
@pytest.mark.parametrize("arrays, basis", [({"Fiber": [1, 2, 2]}, BASIS_FROM_FIBER),
                                           (PLAIN_FULL, BASIS_FROM_FULL)], ids=["fiber", "full"])
def test_plain_arrays_write_one_basis_repeated_per_point(tmp_path, monkeypatch, t10, arrays, basis):
    prefix = _convert(tmp_path, monkeypatch, t10, arrays)
    points = 5 if t10 else 1
    assert prefix.with_suffix(".bases").read_text() == f"1 {points}\n1 {' '.join([basis] * points)}\n"


@pytest.mark.parametrize("plain", [{}, PLAIN_FULL], ids=["alone", "beside_plain"])
def test_per_point_arrays_write_five_bases_centroid_first(tmp_path, monkeypatch, plain):
    prefix = _convert(tmp_path, monkeypatch, True, {**plain, **_per_point_arrays()})
    header, line = prefix.with_suffix(".bases").read_text().splitlines()
    assert header == "1 5"
    expected = [1.0] + [x for perm in AXIS_PERMUTATIONS for axis in perm for x in np.eye(3)[axis]]
    assert np.array_equal(np.array(line.split(), float), expected)


def test_per_point_arrays_on_t4_raise(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="T10"):
        _convert(tmp_path, monkeypatch, False, _per_point_arrays())


@pytest.mark.parametrize("missing", ["Fiber_4", "Sheet_0", "Sheetnormal_2"])
def test_partial_per_point_arrays_raise(tmp_path, monkeypatch, missing):
    arrays = _per_point_arrays()
    del arrays[missing]
    with pytest.raises(ValueError, match=missing):
        _convert(tmp_path, monkeypatch, True, arrays)


def test_scale_multiplies_node_coordinates(tmp_path, monkeypatch):
    prefix = _convert(tmp_path, monkeypatch, False, PLAIN_FULL, "-scale", "2")
    assert prefix.with_suffix(".node").read_text() == (
        "4 3 1 0\n1 0.0 0.0 0.0 0\n2 2.0 0.0 0.0 0\n3 0.0 2.0 0.0 0\n4 0.0 0.0 2.0 0\n")

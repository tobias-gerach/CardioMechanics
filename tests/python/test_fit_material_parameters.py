"""Unit tests for FitMaterialParameters: the optimizer's inputs and its arithmetic.

What the optimizer reads before it runs anything - the settings file it edits, the material law and
the parameter names that law's stiffness level and exponents live under, and the cavity volume of the
mesh the settings file names - and the decisions it then makes: the Klotz end-diastolic
pressure-volume relation, the exponential fit, the parameter scalings and where the loop stands.

No binary, no mesh generation, no gmsh: the meshes here are written by hand, the only geometry with a
volume to check is one whose volume is known in closed form, and the arithmetic is checked against
published anchors and against curves generated from known parameters.
"""
import numpy as np
import pytest

import FitMaterialParameters as fmp

SETTINGS = """<?xml version="1.0" encoding="UTF-8"?>
<settings>Reference recovery</settings>

<Mesh>
    <Type>T10</Type>
    <Tetgen>
        <Unit>1e-3</Unit>
        <Nodes>./tetgen/ellipsoid.node</Nodes>
        <Elements>./tetgen/ellipsoid.ele</Elements>
        <Surfaces>./tetgen/ellipsoid.sur</Surfaces>
    </Tetgen>
</Mesh>

<Materials>
    <!-- Usyk et al. (2002), with a bulk modulus this geometry needs -->
    <Mat_30>
        <Type>Usyk</Type>
        <Usyk><a>88.0</a><bff>5.0</bff><bss>6.0</bss><bnn>3.0</bnn>
              <bfs>12.0</bfs><bfn>2.0</bfn><bns>2.0</bns><k>1e7</k></Usyk>
    </Mat_30>
</Materials>
"""

# The endocardium of the Land et al. (2015) ellipsoid, in mm: a prolate spheroid of equatorial
# radius A and polar radius C, truncated by the base plane at Z_BASE.
A, C, Z_BASE = 7.0, 17.0, 5.0
# Integrating pi * A^2 * (1 - z^2 / C^2) from the apex at -C to the base plane.
CAVITY_ML = 1e-3 * np.pi * A**2 * ((Z_BASE + C) - (Z_BASE**3 + C**3) / (3 * C**2))


@pytest.fixture
def settings_file(tmp_path):
    path = tmp_path / "recovery.xml"
    path.write_text(SETTINGS)
    return path


def write_mesh(directory, nodes, faces):
    """Write nodes (n, 3) and faces, a list of (node indices from 1, material, surface), as the
    tetgen .node and .sur pair of a mesh named `mesh`."""
    with open(directory / "mesh.node", "w") as out:
        out.write(f"{len(nodes)} 3 1 0\n")
        for n, (x, y, z) in enumerate(nodes, 1):
            out.write(f"{n} {x:.17g} {y:.17g} {z:.17g} 0\n")
    with open(directory / "mesh.sur", "w") as out:
        out.write(f"{len(faces)} {len(faces[0][0])} 2\n")
        for n, (face, material, surface) in enumerate(faces, 1):
            out.write(f"{n} {' '.join(map(str, face))} {material} {surface}\n")


def truncated_ellipsoid(slices, sectors):
    """A triangulation of the truncated-ellipsoid cavity of CAVITY_ML, closed by a lid on its base
    plane. The faces are wound so that their normals point out of the cavity, as a cavity surface of
    the solver is. Inscribed in the smooth surface, so the volume it encloses falls short of the
    closed-form one by the discretization error."""
    z = np.linspace(-C, Z_BASE, slices + 1)[1:]        # the apex is a single node, added below
    theta = np.linspace(0, 2 * np.pi, sectors, endpoint=False)
    radius = A * np.sqrt(1 - (z / C) ** 2)
    ring = np.stack([np.outer(radius, np.cos(theta)), np.outer(radius, np.sin(theta)),
                     np.repeat(z[:, None], sectors, axis=1)], axis=-1).reshape(-1, 3)
    nodes = np.vstack([[[0.0, 0.0, -C]], ring, [[0.0, 0.0, Z_BASE]]])
    apex, centre = 1, len(nodes)

    def node(s, k):                                     # node of slice s at sector k, numbered from 1
        return 2 + s * sectors + k % sectors

    faces = [([apex, node(0, k + 1), node(0, k)], 1, 1) for k in range(sectors)]
    for s in range(slices - 1):
        for k in range(sectors):
            a, b, c, d = node(s, k), node(s, k + 1), node(s + 1, k + 1), node(s + 1, k)
            faces += [([a, b, c], 1, 1), ([a, c, d], 1, 1)]
    # The lid carries the cavity's surface index and a material index of its own, the handle that
    # keeps it out of the pressure load.
    faces += [([centre, node(slices - 1, k), node(slices - 1, k + 1)], 4, 1) for k in range(sectors)]
    return nodes, faces


def test_settings_round_trip_is_unchanged(settings_file, tmp_path):
    out = tmp_path / "written.xml"
    fmp.write_settings(fmp.read_settings(settings_file), out)
    assert out.read_text() == SETTINGS


def test_changed_parameter_survives_a_round_trip(settings_file, tmp_path):
    out = tmp_path / "written.xml"
    root = fmp.read_settings(settings_file)
    fmp.set_parameter(root, "Materials.Mat_30.Usyk.a", 176.0)
    fmp.write_settings(root, out)

    assert fmp.get_parameter(fmp.read_settings(out), "Materials.Mat_30.Usyk.a") == "176.0"
    assert out.read_text() == SETTINGS.replace("<a>88.0</a>", "<a>176.0</a>")


def test_an_empty_element_is_written_with_a_closing_tag(tmp_path):
    """The solver's reader knows no self-closing tag: it would take <Bases /> for a tag named
    "Bases/" and then reject the closing tag it never opened."""
    text = SETTINGS.replace("<Type>T10</Type>", "<Type>T10</Type>\n    <Bases></Bases>")
    (tmp_path / "empty.xml").write_text(text)
    out = tmp_path / "written.xml"
    fmp.write_settings(fmp.read_settings(tmp_path / "empty.xml"), out)
    assert out.read_text() == text


def test_missing_parameter_raises_rather_than_being_created(settings_file):
    root = fmp.read_settings(settings_file)
    with pytest.raises(KeyError):
        fmp.set_parameter(root, "Materials.Mat_30.Usyk.aScale", 2.0)
    with pytest.raises(KeyError):
        fmp.get_parameter(root, "Materials.Mat_30.Usyk.aScale")
    assert fmp.get_parameter(root, "Materials.Mat_30.Usyk.aScale", "1.0") == "1.0"


def test_material_law_is_read_from_the_file(settings_file):
    law, prefixes = fmp.material_law(fmp.read_settings(settings_file))
    assert law == "Usyk"
    assert prefixes == ["Materials.Mat_30.Usyk"]


def test_every_material_of_the_law_is_reported(settings_file, tmp_path):
    second = SETTINGS.replace("</Materials>", "<Mat_31><Type>Usyk</Type></Mat_31></Materials>")
    (tmp_path / "two.xml").write_text(second)
    law, prefixes = fmp.material_law(fmp.read_settings(tmp_path / "two.xml"))
    assert law == "Usyk"
    assert prefixes == ["Materials.Mat_30.Usyk", "Materials.Mat_31.Usyk"]


def test_materials_of_different_laws_raise(tmp_path):
    mixed = SETTINGS.replace("</Materials>", "<Mat_31><Type>Guccione</Type></Mat_31></Materials>")
    (tmp_path / "mixed.xml").write_text(mixed)
    with pytest.raises(ValueError, match="Guccione"):
        fmp.material_law(fmp.read_settings(tmp_path / "mixed.xml"))


@pytest.mark.parametrize("law,stiffness,exponents", [
    ("Usyk", ("a",), ("bff", "bss", "bnn", "bfs", "bfn", "bns")),
    ("Guccione", ("C",), ("bf", "bt", "bfs")),
    ("Holzapfel", ("a", "af", "as", "afs"), ("b", "bf", "bs", "bfs")),
])
def test_law_parameter_table(law, stiffness, exponents):
    assert fmp.law_parameters(law) == (stiffness, exponents)


def test_bulk_modulus_is_not_a_scaled_parameter():
    for law, bulk in [("Usyk", "k"), ("Guccione", "K"), ("Holzapfel", "k")]:
        stiffness, exponents = fmp.law_parameters(law)
        assert bulk not in stiffness and bulk not in exponents


def test_parameter_names_can_be_overridden():
    assert fmp.law_parameters("Usyk", stiffness=["a", "aScale"]) == (("a", "aScale"),
                                                                    fmp.LAW_PARAMETERS["Usyk"][1])
    assert fmp.law_parameters("NeoHooke", stiffness=["c1"], exponents=[]) == (("c1",), ())


def test_an_unlisted_law_without_overrides_raises():
    with pytest.raises(ValueError, match="NeoHooke"):
        fmp.law_parameters("NeoHooke")


def test_mesh_files_are_resolved_against_the_settings_file(settings_file):
    nodes, surfaces, unit = fmp.mesh_files(fmp.read_settings(settings_file), settings_file)
    assert nodes == settings_file.parent / "tetgen" / "ellipsoid.node"
    assert surfaces == settings_file.parent / "tetgen" / "ellipsoid.sur"
    assert unit == 1e-3


def test_cavity_volume_matches_the_truncated_ellipsoid(tmp_path):
    write_mesh(tmp_path, *truncated_ellipsoid(slices=80, sectors=80))
    volume = fmp.cavity_volume(tmp_path / "mesh.node", tmp_path / "mesh.sur", 1, 1e-3)
    assert volume == pytest.approx(CAVITY_ML, rel=2e-3)
    assert volume < CAVITY_ML                       # the triangulation is inscribed


def test_cavity_volume_ignores_the_other_surfaces(tmp_path):
    nodes, faces = truncated_ellipsoid(slices=20, sectors=20)
    write_mesh(tmp_path, nodes, faces)
    cavity = fmp.cavity_volume(tmp_path / "mesh.node", tmp_path / "mesh.sur", 1, 1e-3)

    others = [(face, 1, 2) for face, _, _ in faces[:10]]               # an open patch of epicardium
    write_mesh(tmp_path, nodes, faces + others)
    assert fmp.cavity_volume(tmp_path / "mesh.node", tmp_path / "mesh.sur", 1, 1e-3) == cavity


def test_cavity_volume_of_six_node_faces(tmp_path):
    """A tetrahedron, whose faces are flat, so its mid-edge nodes sit at the edge midpoints and its
    volume is the closed-form one. The six-node face is spanned as the hexagon through all six of
    its nodes, which is the flat triangle here."""
    corners = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    midpoints = {e: 5 + k for k, e in enumerate(edges)}                 # node numbers from 1
    nodes = np.vstack([corners] + [(corners[i] + corners[j]) / 2 for i, j in edges])

    def face(a, b, c):
        mid = [midpoints[tuple(sorted(e))] for e in [(a, b), (b, c), (c, a)]]
        return ([a + 1, b + 1, c + 1] + mid, 1, 1)

    # Wound to point out of the tetrahedron, the orientation of a cavity surface.
    write_mesh(tmp_path, nodes, [face(0, 2, 1), face(0, 1, 3), face(0, 3, 2), face(1, 2, 3)])
    volume = abs(np.linalg.det(corners[1:] - corners[0])) / 6
    assert fmp.cavity_volume(tmp_path / "mesh.node", tmp_path / "mesh.sur", 1, 1e-3) == pytest.approx(
        1e-3 * volume)


def test_cavity_volume_with_a_traction_scaling_attribute(tmp_path):
    """A surface file may carry a third attribute, the traction scaling, behind the surface index,
    so the surface index is neither the last column of every mesh nor at a fixed one."""
    write_mesh(tmp_path, *truncated_ellipsoid(slices=20, sectors=20))
    surfaces = tmp_path / "mesh.sur"
    volume = fmp.cavity_volume(tmp_path / "mesh.node", surfaces, 1, 1e-3)

    header, *rows = surfaces.read_text().splitlines()
    count, per_face, _ = header.split()
    surfaces.write_text("\n".join([f"{count} {per_face} 3"] + [f"{row} 5" for row in rows]) + "\n")
    assert fmp.cavity_volume(tmp_path / "mesh.node", surfaces, 1, 1e-3) == volume


def test_an_open_cavity_surface_raises(tmp_path):
    nodes, faces = truncated_ellipsoid(slices=20, sectors=20)
    write_mesh(tmp_path, nodes, faces[:-1])                            # a hole where one lid face was
    with pytest.raises(AssertionError):
        fmp.cavity_volume(tmp_path / "mesh.node", tmp_path / "mesh.sur", 1, 1e-3)


# A measured end-diastolic pair of the size the Klotz relation was established on, and the one the
# example works at: 8 mmHg in a ventricle of about 121 ml.
P_MEASURED, V_MEASURED = 8.0, 121.0


def test_klotz_returns_the_measured_pressure_at_the_measured_volume():
    unloaded, v30 = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    assert fmp.klotz_pressure(V_MEASURED, unloaded, v30) == pytest.approx(P_MEASURED)


def test_klotz_unloaded_volume_is_the_published_fraction():
    unloaded, v30 = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    assert unloaded == pytest.approx(V_MEASURED * (0.6 - 0.006 * P_MEASURED))
    assert unloaded / V_MEASURED == pytest.approx(0.552)     # about 55 % at 8 mmHg
    assert fmp.klotz_pressure(unloaded, unloaded, v30) == pytest.approx(0.0)


def test_klotz_v30_satisfies_its_defining_relation():
    """V30 is where the normalized curve reaches its prefactor, the published estimate of the volume
    at 30 mmHg, and is fixed by the measured pair through (Vm - V0) / (V30 - V0) = (Pm / An)^(1/Bn)."""
    unloaded, v30 = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    assert fmp.klotz_pressure(v30, unloaded, v30) == pytest.approx(fmp.KLOTZ_AN)
    assert (V_MEASURED - unloaded) / (v30 - unloaded) == pytest.approx(
        (P_MEASURED / fmp.KLOTZ_AN) ** (1 / fmp.KLOTZ_BN))


def test_klotz_unloaded_volume_is_scale_free():
    """The relation's unloaded-volume term is a fraction of the measured volume, so a chamber of
    twice the size has twice the unloaded volume at the same pressure."""
    for volume in (V_MEASURED, 2 * V_MEASURED):
        assert fmp.klotz_volumes(P_MEASURED, volume)[0] / volume == pytest.approx(0.552)


def test_klotz_curve_spans_the_inflation():
    volumes, pressures = fmp.klotz_curve(P_MEASURED, V_MEASURED, samples=32)
    unloaded, _ = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    assert (len(volumes), len(pressures)) == (32, 32)
    assert volumes[0] == pytest.approx(unloaded) and volumes[-1] == pytest.approx(V_MEASURED)
    assert pressures[0] == pytest.approx(0.0) and pressures[-1] == pytest.approx(P_MEASURED)
    assert np.all(np.diff(pressures) > 0)


@pytest.mark.parametrize("prefactor,exponent", [(0.5, 0.04), (2.5, 0.01), (0.05, 0.12)])
def test_the_exponential_fit_recovers_known_parameters(prefactor, exponent):
    unloaded = 60.0
    volumes = np.linspace(unloaded, 120.0, 40)
    pressures = fmp.exponential_pressure(volumes, unloaded, prefactor, exponent)
    assert fmp.fit_exponential(volumes, pressures, unloaded) == pytest.approx(
        (prefactor, exponent), rel=1e-6)


def test_the_exponential_model_is_anchored_at_the_unloaded_volume():
    assert fmp.exponential_pressure(60.0, 60.0, 0.5, 0.04) == pytest.approx(0.0)


def test_the_klotz_curve_is_fitted_by_the_exponential_model():
    """The fit the optimizer compares against: the exponential model over the Klotz relation of the
    measured pair reproduces that pair's pressure."""
    volumes, pressures = fmp.klotz_curve(P_MEASURED, V_MEASURED)
    unloaded, _ = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    prefactor, exponent = fmp.fit_exponential(volumes, pressures, unloaded)
    assert exponent > 0 and prefactor > 0
    assert fmp.exponential_pressure(V_MEASURED, unloaded, prefactor, exponent) == pytest.approx(
        P_MEASURED, rel=0.05)


def test_the_scalings_are_the_ratios_of_the_two_fits():
    scalings = fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=(0.3, 0.04))
    assert (scalings.stiffness, scalings.exponent) == pytest.approx((2.0, 1.5))
    assert scalings.clamped == ()


@pytest.mark.parametrize("simulated_fit,expected", [
    ((0.001, 0.001), fmp.SCALING_BOUNDS[1]),
    ((100.0, 100.0), fmp.SCALING_BOUNDS[0]),
])
def test_both_bounds_of_the_scaling_interval(simulated_fit, expected):
    scalings = fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=simulated_fit)
    assert (scalings.stiffness, scalings.exponent) == pytest.approx((expected, expected))
    assert scalings.clamped == ("stiffness", "exponent")


def test_only_the_scaling_that_left_the_interval_is_reported_as_clamped():
    scalings = fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=(0.3, 0.001))
    assert scalings.clamped == ("exponent",)
    assert (scalings.stiffness, scalings.exponent) == pytest.approx((2.0, fmp.SCALING_BOUNDS[1]))


def test_the_magnitude_of_the_simulated_exponent_is_used():
    """A simulated curve flat enough for the fit to return a negative exponent still stiffens the
    law rather than inverting it."""
    negative = fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=(0.3, -0.04))
    assert negative.exponent == pytest.approx(1.5)
    assert negative == fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=(0.3, 0.04))


UNCHANGED = fmp.Scalings(stiffness=1.0, exponent=1.0, clamped=())
MOVING = fmp.Scalings(stiffness=1.5, exponent=1.2, clamped=())


def test_convergence_when_both_volumes_are_within_tolerance():
    state = fmp.convergence(unloaded=66.8, klotz_unloaded=66.8, end_diastolic=121.0,
                            measured=V_MEASURED, scalings=MOVING)
    assert state.converged and not state.stagnated
    assert (state.unloaded, state.end_diastolic, state.stagnation) == pytest.approx((0.0, 0.0, 0.5))


def test_the_unloaded_volume_residual_binds():
    state = fmp.convergence(unloaded=80.0, klotz_unloaded=66.8, end_diastolic=121.0,
                            measured=V_MEASURED, scalings=MOVING)
    assert not state.converged and not state.stagnated
    assert state.unloaded == pytest.approx(13.2 / 66.8)
    assert state.end_diastolic == pytest.approx(0.0)


def test_the_end_diastolic_volume_residual_binds():
    """Near-tautological, since the recovery drives the loaded configuration onto the target; it
    binds only when the inner loop did not get there, which is what it is reported for."""
    state = fmp.convergence(unloaded=66.8, klotz_unloaded=66.8, end_diastolic=110.0,
                            measured=V_MEASURED, scalings=MOVING)
    assert not state.converged and not state.stagnated
    assert state.end_diastolic == pytest.approx(11.0 / 121.0)


def test_only_stagnation_fires():
    state = fmp.convergence(unloaded=80.0, klotz_unloaded=66.8, end_diastolic=110.0,
                            measured=V_MEASURED, scalings=UNCHANGED)
    assert state.stagnated and not state.converged
    assert state.stagnation == pytest.approx(0.0)


def test_a_scaling_just_outside_the_stagnation_tolerance_keeps_the_loop_running():
    moved = fmp.Scalings(stiffness=1.0, exponent=1.0 + 2 * fmp.STAGNATION_TOLERANCE, clamped=())
    assert not fmp.convergence(unloaded=80.0, klotz_unloaded=66.8, end_diastolic=110.0,
                               measured=V_MEASURED, scalings=moved).stagnated


def test_a_measured_pair_that_is_not_end_diastolic_raises():
    with pytest.raises(AssertionError):
        fmp.klotz_volumes(0.0, V_MEASURED)


def test_the_klotz_relation_below_the_unloaded_volume_raises():
    """A volume below the unloaded one is on no end-diastolic branch; the fractional exponent would
    otherwise return a nan that only surfaces once it has travelled into a fit."""
    unloaded, v30 = fmp.klotz_volumes(P_MEASURED, V_MEASURED)
    with pytest.raises(AssertionError):
        fmp.klotz_pressure(unloaded - 1.0, unloaded, v30)


def test_a_simulated_fit_without_a_positive_prefactor_raises():
    """A negative prefactor is a fit that found no rising curve at all, not one asking for a large
    move, so it is not something the clamp should absorb."""
    with pytest.raises(AssertionError):
        fmp.parameter_scalings(klotz_fit=(0.6, 0.06), simulated_fit=(-0.3, 0.04))

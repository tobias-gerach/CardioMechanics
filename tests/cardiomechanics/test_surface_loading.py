"""Six-node surface faces declared as three-node elements, which the tetgen loader refines.

A surface of a quadratic mesh can be written as six-node faces or as the three-node triangles each
of those faces refines into. Declaring a surface of six-node faces T3 asks the loader for the
second from the first, and the two have to load to the same mesh: the loader's refinement is the
one the surface file would have been written with.

The cantilever of test_p2p1 carries the pressure on its T6 top surface and, in the Robin fixture,
a Robin boundary on its free end. Both are run here with that surface declared T3.
"""
from pathlib import Path

import numpy as np
import pytest

from helpers.cantilever import SURFACE, write_mesh
from helpers.compare import read_vtu_points
from helpers.run import run_binary

FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever.xml"
ROBIN_FIXTURE = Path(__file__).parent / "fixtures" / "p2p1_cantilever_robin.xml"

# The four triangles a six-node face refines into, in the node order of the face, as
# CBModelLoaderTetgen::LoadSurfaces writes them.
SUB_TRIANGLES = ((0, 3, 5), (3, 1, 4), (3, 4, 5), (5, 4, 2))

# The final tip displacement is 2e-5 of a block of unit height, so a relative tolerance on the
# coordinates would be dominated by the undeformed geometry. Both runs solve the same equations to
# the fixture's 1e-10, and differ only in the order the surface elements are assembled in.
ATOL = 1e-12


def _refine(surfaces, index):
    """Rewrite the tetgen surface file in place, replacing every six-node face of surface `index`
    by the four three-node faces it refines into. Faces of other surfaces are left alone."""
    lines = surfaces.read_text().split("\n")
    header = lines[0].split()
    rows = [line.split() for line in lines[1:] if line.strip()]
    refined = []
    for row in rows:
        nodes, attributes = row[1:-int(header[2])], row[-int(header[2]):]
        if int(attributes[1]) == index and len(nodes) == 6:
            refined += [[nodes[i] for i in triangle] + attributes for triangle in SUB_TRIANGLES]
        else:
            refined.append(nodes + attributes)
    header[0] = str(len(refined))
    surfaces.write_text("\n".join([" ".join(header)]
                                  + [" ".join([str(n)] + row) for n, row in enumerate(refined, 1)]) + "\n")


def _run(binary, cm_env, wd, fixture, refine=False, quadratic_end=False):
    """Stage the cantilever and `fixture` into wd with the pressure surface declared T3, refining
    that surface into three-node faces first if asked, and return the exported VTU directory."""
    (wd / "tetgen").mkdir(parents=True)
    (wd / "Results").mkdir()
    write_mesh(wd / "tetgen", quadratic_end=quadratic_end)
    if refine:
        _refine(wd / "tetgen" / "cantilever.sur", SURFACE)
    text = fixture.read_text()
    declaration = f"<Surface_{SURFACE}><Type>T6</Type></Surface_{SURFACE}>"
    assert text.count(declaration) == 1, f"{fixture.name}: cannot substitute {declaration}"
    (wd / fixture.name).write_text(
        text.replace(declaration, f"<Surface_{SURFACE}><Type>T3</Type></Surface_{SURFACE}>"))
    run_binary(binary("CardioMechanics"), ["-settings", fixture.name], cwd=wd, env=cm_env)
    return wd / "Results" / "cantilever_vtu"


def _final_points(vtu_dir):
    last = max(vtu_dir.glob("cantilever.*.vtu"), key=lambda p: int(p.stem.split(".")[1]))
    return read_vtu_points(last)[1]


def test_six_node_faces_declared_t3_match_the_triangles_they_refine_into(binary, cm_env, tmp_path):
    """The pressure-loaded top surface, once as six-node faces and once written out as the four
    three-node faces each of them refines into. A loader that stores one triangle of every face
    four times leaves three quarters of the surface unloaded."""
    loader = _final_points(_run(binary, cm_env, tmp_path / "loader", FIXTURE))
    written = _final_points(_run(binary, cm_env, tmp_path / "written", FIXTURE, refine=True))
    assert loader.shape == written.shape, f"{loader.shape} nodes against {written.shape}"
    assert np.allclose(loader, written, rtol=0, atol=ATOL), \
        f"largest coordinate difference {np.abs(loader - written).max():.3e} m"


@pytest.mark.parametrize("fixture", [FIXTURE, ROBIN_FIXTURE], ids=["pressure", "robin"])
def test_surfaces_of_six_node_faces_keep_their_element_indices(binary, cm_env, tmp_path, fixture):
    """Every refined triangle is an element of its own, so the indices the elements carry stay in
    step with the container CBModel::CheckElementsIndexes() walks. The Robin fixture declares the
    free end CONTACT_ROBIN, which is a three-node element, and gets it as six-node faces here."""
    _run(binary, cm_env, tmp_path, fixture, quadratic_end=True)

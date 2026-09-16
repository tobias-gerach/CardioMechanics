"""Truncated ellipsoid of the Land et al. (2015) benchmark problems 2 and 3, meshed by gmsh as
tetrahedra and carrying the analytic fibre field of problem 3.

Run from tests/ as `python -m helpers.ellipsoid DIRECTORY --level K` to write one mesh of the family.
"""
import numpy as np

from helpers.gmsh_tetgen import LAYOUTS, write_tetgen

MATERIAL = 30                              # physical tag of the volume
ENDO, EPI, BASE = 1, 2, 3                  # physical tags of the surfaces, as in examples/benchmark2015
ENDO_APEX, EPI_APEX = (0.0, 0.0, -17.0), (0.0, 0.0, -20.0)
Z_BASE = 5.0
# Element sizes of level 0 at the epicardial apex and at the base, in mm. The apex carries the fibre
# singularity and both quantities of interest of problems 2 and 3, so the mesh is finest there.
SIZE_APEX, SIZE_BASE = 1.0, 2.0
# The centroid, then the points of the 4-point rule, as barycentric coordinates in the vertex order
# of the code. A line of the bases file holds the frame at each of them, in this order.
ALPHA, BETA = (5 + 3 * np.sqrt(5)) / 20, (5 - np.sqrt(5)) / 20
BASIS_POINTS = np.array([[0.25] * 4] + [[ALPHA if i == q else BETA for i in range(4)] for q in range(4)])


def fibre_frames(X):
    """Fibre, sheet and sheet-normal directions, as rows of (n, 3, 3), at the points X (n, 3) in mm.

    This is the fibre field of problem 3, as in xyz_to_fiber.m of the benchmark repository: the
    transmural coordinate t, 0 on the endocardium and 1 on the epicardium, fixes the ellipsoidal
    surface through the point, r_s = 7 + 3t and r_l = 17 + 3t, and the fibre angle alpha = 90 - 180t
    from the circumferential direction dX/dv towards the meridian dX/du. The problem's Guccione law is
    transversely isotropic, so the sheet only completes the frame; it lies in that surface.
    """
    x, y, z = X.T

    def outside(t):                        # falls monotonically in t
        return (x**2 + y**2) / (7 + 3 * t)**2 + z**2 / (17 + 3 * t)**2 - 1

    assert np.all((outside(0.0) >= -1e-9) & (outside(1.0) <= 1e-9)), "a point lies outside the wall"
    lo, hi = np.zeros(len(X)), np.ones(len(X))
    for _ in range(60):                    # bisection to round-off
        t = (lo + hi) / 2
        out = outside(t) > 0
        lo, hi = np.where(out, t, lo), np.where(out, hi, t)

    r_s, r_l = 7 + 3 * t, 17 + 3 * t
    u = -np.arccos(np.clip(z / r_l, -1, 1))           # in [-pi, 0], the apex at -pi
    v = np.arctan2(-y, -x)                             # sin u <= 0, so x = r_s sin u cos v
    du = np.stack([r_s * np.cos(u) * np.cos(v), r_s * np.cos(u) * np.sin(v), -r_l * np.sin(u)], axis=1)
    dv = np.stack([-r_s * np.sin(u) * np.sin(v), r_s * np.sin(u) * np.cos(v), np.zeros_like(u)], axis=1)
    assert np.all(np.linalg.norm(dv, axis=1) > 0), "a point lies on the long axis, where the fibres are undefined"
    du /= np.linalg.norm(du, axis=1, keepdims=True)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    alpha = np.radians(90 - 180 * t)[:, None]
    fibre = du * np.sin(alpha) + dv * np.cos(alpha)
    sheet = du * np.cos(alpha) - dv * np.sin(alpha)
    return np.stack([fibre, sheet, np.cross(fibre, sheet)], axis=1)


def write_ellipsoid(directory, level, order=2, curved=True):
    """Write the ellipsoid at refinement level `level` as tetgen ellipsoid.node, .ele, .sur and .bases
    into directory, in mm. The element size grows linearly in z from SIZE_APEX at the apex to
    SIZE_BASE at the base, both divided by sqrt(2) per level, which about triples the elements. The
    base nodes are fixed in all directions. The endocardium, epicardium and base are written as faces
    of the element order with surface indices ENDO, EPI and BASE. Returns the numbers of elements and
    nodes.

    order 1 writes T4 elements, order 2 T10 elements. With curved=False the mid-edge nodes stay at
    the edge midpoints instead of moving onto the curved boundary, so the T10 mesh covers exactly the
    domain of the T4 mesh of the same level: comparing element types on the two then measures the
    discretization alone, not the geometry."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ

        def ellipsoid(r_s, r_l):
            tag = occ.addSphere(0, 0, 0, 1)
            occ.dilate([(3, tag)], 0, 0, 0, r_s, r_s, r_l)
            return [(3, tag)]

        shell, _ = occ.cut(ellipsoid(10, 20), ellipsoid(7, 17))
        shell, _ = occ.cut(shell, [(3, occ.addBox(-20, -20, Z_BASE, 40, 40, 20))])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in shell], MATERIAL)
        surfaces = {ENDO: [], EPI: [], BASE: []}
        for _, tag in gmsh.model.getEntities(2):
            if gmsh.model.getType(2, tag) == "Plane":
                surfaces[BASE].append(tag)
            else:
                surfaces[ENDO if gmsh.model.getBoundingBox(2, tag)[2] > -18.5 else EPI].append(tag)
        for group, tags in surfaces.items():
            gmsh.model.addPhysicalGroup(2, tags, group)

        scale = 2 ** (-level / 2)
        slope = scale * (SIZE_BASE - SIZE_APEX) / (Z_BASE - EPI_APEX[2])
        size = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(size, "F", f"{scale * SIZE_APEX - slope * EPI_APEX[2]} + {slope} * z")
        gmsh.model.mesh.field.setAsBackgroundMesh(size)
        for option in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(option, 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        if order == 2:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0 if curved else 1)
            gmsh.model.mesh.setOrder(2)

        # The quantities of interest are the displacements of the two apex points, read at nodes.
        coords = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
        for apex in (ENDO_APEX, EPI_APEX):
            assert np.linalg.norm(coords - apex, axis=1).min() < 1e-9, f"no node at the apex {apex}"

        write_tetgen(directory, "ellipsoid", list(surfaces), {BASE: 7})

        # Same element order as write_tetgen. gmsh's reference coordinates are the barycentric
        # coordinates of vertices 2 to 4.
        frames = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, MATERIAL):
            _, det, points = gmsh.model.mesh.getJacobians(LAYOUTS[order][0], BASIS_POINTS[:, 1:].ravel(), entity)
            assert np.all(det > 0), "an element is inverted at a basis point"
            frames.append(fibre_frames(points.reshape(-1, 3)).reshape(-1, 9 * len(BASIS_POINTS)))
        frames = np.vstack(frames)
        with open(directory / "ellipsoid.bases", "w") as out:
            out.write(f"{len(frames)} {len(BASIS_POINTS)}\n")
            for n, row in enumerate(frames, 1):
                out.write(f"{n} {' '.join(f'{a:.17g}' for a in row)}\n")
        return len(frames), len(coords)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--level", type=int, default=0, help="refinement level, 0 is the coarsest")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    elements, nodes = write_ellipsoid(args.directory, args.level)
    print(f"level {args.level}: {elements} elements, {nodes} nodes")

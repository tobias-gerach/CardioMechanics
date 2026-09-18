"""Truncated ellipsoid of the Land et al. (2015) benchmark problems 2 and 3, meshed by gmsh as
tetrahedra and carrying the analytic fibre field of problem 3.

Run from tests/ as `python -m helpers.ellipsoid DIRECTORY --level K` to write one mesh of the family.
"""
import numpy as np

from helpers.gmsh_tetgen import LAYOUTS, TRIANGLE3, closed_surface, write_tetgen

MATERIAL = 30                              # physical tag of the volume
ENDO, EPI, BASE = 1, 2, 3                  # physical tags of the surfaces, as in examples/benchmark2015
LID = 4                                    # physical tag of the lid, which is its material index
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


def write_ellipsoid(directory, level, order=2, curved=True, lid=False, scale=1.0):
    """Write the ellipsoid at refinement level `level` as tetgen ellipsoid.node, .ele, .sur and .bases
    into directory, in mm. The element size grows linearly in z from SIZE_APEX at the apex to
    SIZE_BASE at the base, both divided by sqrt(2) per level, which about triples the elements. The
    base nodes are fixed in all directions. The endocardium, epicardium and base are written as faces
    of the element order with surface indices ENDO, EPI and BASE. Returns the numbers of elements and
    nodes.

    order 1 writes T4 elements, order 2 T10 elements. With curved=False the mid-edge nodes stay at
    the edge midpoints instead of moving onto the curved boundary, so the T10 mesh covers exactly the
    domain of the T4 mesh of the same level: comparing element types on the two then measures the
    discretization alone, not the geometry.

    With lid=True the base opening of the endocardium is closed by a plane surface on its base curve
    loop, so the enclosed volume of the endocardium is a true cavity volume instead of a number that
    depends on where the origin sits. Its faces carry surface index ENDO, since cavity membership is
    by surface index, and material index LID, the handle a cavity plugin's IgnoredSurfaceIndices uses
    to keep them out of the pressure load. Its interior nodes lie in no tetrahedron and are fixed in
    all directions: the solver adds a diagonal 1.0 on every constrained component onto the Jacobian
    independently of the element assembly, so such a node gets an identity row instead of an empty one.

    The endocardium and the lid then point out of the cavity rather than out of the wall, so the
    cavity encloses a positive volume, as every other cavity mesh of the repository does. The sign of
    the pressure that inflates the cavity turns over with them.

    `scale` dilates the node coordinates uniformly after meshing, leaving element count and topology
    those of the unscaled mesh of the same level. Only a uniform dilation keeps the analytic fibre
    field valid, since it is defined by the hard-coded endocardial and epicardial radii; its frames
    are unit vectors and are unchanged."""
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
        if lid:
            # Building the lid on the curves the endocardium and the base already share makes the
            # closure conforming: gmsh meshes a curve once, whatever surfaces bound it.
            def curves(group):
                return {abs(tag) for _, tag in gmsh.model.getBoundary([(2, t) for t in surfaces[group]],
                                                                     combined=False, oriented=False)}
            loop = sorted(curves(ENDO) & curves(BASE))
            assert loop, "the endocardium and the base share no curve to close the cavity on"
            surfaces[LID] = [occ.addPlaneSurface([occ.addCurveLoop(loop)])]
            occ.synchronize()
        for group, tags in surfaces.items():
            gmsh.model.addPhysicalGroup(2, tags, group)

        mesh_scale = 2 ** (-level / 2)
        slope = mesh_scale * (SIZE_BASE - SIZE_APEX) / (Z_BASE - EPI_APEX[2])
        size = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(size, "F", f"{mesh_scale * SIZE_APEX - slope * EPI_APEX[2]} + {slope} * z")
        gmsh.model.mesh.field.setAsBackgroundMesh(size)
        for option in ("Mesh.MeshSizeFromPoints", "Mesh.MeshSizeFromCurvature", "Mesh.MeshSizeExtendFromBoundary"):
            gmsh.option.setNumber(option, 0)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize("Netgen")
        if lid:
            # The cavity surface points out of the cavity, so that it encloses a positive volume. The
            # cavity lies below the base, so the lid closes it consistently only if it points upwards.
            face = gmsh.model.mesh.getElementsByType(TRIANGLE3, surfaces[LID][0])[1][:3]
            a, b, c = (np.array(gmsh.model.mesh.getNode(node)[0]) for node in face)
            if np.cross(b - a, c - a)[2] < 0:
                gmsh.model.mesh.reverse([(2, surfaces[LID][0])])
        if order == 2:
            gmsh.option.setNumber("Mesh.SecondOrderLinear", 0 if curved else 1)
            gmsh.model.mesh.setOrder(2)

        # The quantities of interest are the displacements of the two apex points, read at nodes.
        coords = gmsh.model.mesh.getNodes()[1].reshape(-1, 3)
        for apex in (ENDO_APEX, EPI_APEX):
            assert np.linalg.norm(coords - apex, axis=1).min() < 1e-9, f"no node at the apex {apex}"

        # Same element order as write_tetgen. gmsh's reference coordinates are the barycentric
        # coordinates of vertices 2 to 4. The frames are read off the unscaled mesh, whose radii the
        # fibre field is written in, and are unit vectors that the dilation leaves alone.
        frames = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, MATERIAL):
            _, det, points = gmsh.model.mesh.getJacobians(LAYOUTS[order][0], BASIS_POINTS[:, 1:].ravel(), entity)
            assert np.all(det > 0), "an element is inverted at a basis point"
            frames.append(fibre_frames(points.reshape(-1, 3)).reshape(-1, 9 * len(BASIS_POINTS)))
        frames = np.vstack(frames)

        if scale != 1.0:
            gmsh.model.mesh.affineTransform([scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0])

        surface_indices = {group: ENDO if group == LID else group for group in surfaces}
        write_tetgen(directory, "ellipsoid", surface_indices, {BASE: 7, LID: 7} if lid else {BASE: 7},
                     inward={ENDO} if lid else ())
        if lid:
            assert closed_surface(directory, "ellipsoid", ENDO), "the lid does not close the cavity"

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
    parser.add_argument("--lid", action="store_true", help="close the endocardial cavity with a lid")
    parser.add_argument("--scale", type=float, default=1.0, help="uniform dilation of the node coordinates")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    elements, nodes = write_ellipsoid(args.directory, args.level, lid=args.lid, scale=args.scale)
    print(f"level {args.level}: {elements} elements, {nodes} nodes")

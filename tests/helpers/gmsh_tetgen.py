"""First- and second-order tetrahedral meshes from gmsh, in the tetgen files and node order of
CardioMechanics.

gmsh places the mid-edge nodes of a second-order mesh on the model geometry, so a curved boundary
is represented to the order of the element instead of by its chords. The T4 or T10 basis functions
at gmsh's quadrature points, in the same node order, integrate over such meshes. Requires gmsh.
"""
import numpy as np

TRIANGLE3, TETRAHEDRON4, TETRAHEDRON10, TRIANGLE6 = 2, 4, 11, 9        # gmsh element types
# CardioMechanics T10 local nodes 9 and 10 sit on the edges (2,4) and (3,4); gmsh numbers them the
# other way round. Vertices, the other mid-edge nodes and six-node triangles agree.
T10_FROM_GMSH = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]
# Per element order: the gmsh tetrahedron and triangle types, the gmsh index of each local node of
# CardioMechanics, and the face node order that reverses a triangle's normal.
LAYOUTS = {1: (TETRAHEDRON4, TRIANGLE3, [0, 1, 2, 3], [0, 2, 1]),
           2: (TETRAHEDRON10, TRIANGLE6, T10_FROM_GMSH, [0, 2, 1, 5, 4, 3])}


def tetrahedron_quadrature(rule, order):
    """gmsh's integration rule on the reference tetrahedron. Returns the weights (q), the shape
    functions of the T4 (order 1) or T10 (order 2) tetrahedron (q, n) and their reference gradients
    (q, n, 3) in the CardioMechanics node order, and the linear shape functions of the vertices
    (q, 4), all at the q points."""
    import gmsh

    element, _, from_gmsh, _ = LAYOUTS[order]
    gmsh.initialize(interruptible=False)
    try:
        points, weights = gmsh.model.mesh.getIntegrationPoints(element, rule)
        N = gmsh.model.mesh.getBasisFunctions(element, points, "Lagrange")[1]
        dN = gmsh.model.mesh.getBasisFunctions(element, points, "GradLagrange")[1]
        L = gmsh.model.mesh.getBasisFunctions(TETRAHEDRON4, points, "Lagrange")[1]
    finally:
        gmsh.finalize()
    q, n = len(weights), len(from_gmsh)
    return weights, N.reshape(q, n)[:, from_gmsh], dN.reshape(q, n, 3)[:, from_gmsh], L.reshape(q, 4)


def write_tetgen(directory, stem, surfaces, fixations):
    """Write the tetrahedra of the current gmsh model, first- or second-order, as tetgen stem.node,
    stem.ele and stem.sur into directory.

    Each element carries the tag of its 3D physical group as material. The triangles of every 2D
    physical group in surfaces, of the order of the tetrahedra and required to lie on the boundary
    of the volume, are written with the group tag as surface index, ordered so that their normals
    point out of the volume. fixations maps a 2D physical group to the fixed components of its nodes
    as a bit mask, 1 for x, 2 for y and 4 for z, combined over the groups a node lies in.
    """
    import gmsh

    (volume,) = gmsh.model.mesh.getElementTypes(3)
    tetrahedron, triangle, from_gmsh, reversed_face = LAYOUTS[gmsh.model.mesh.getElementProperties(volume)[2]]
    assert volume == tetrahedron, f"the volume is meshed with gmsh element type {volume}, not tetrahedra"
    elements, materials = [], []
    for _, group in gmsh.model.getPhysicalGroups(3):
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, group):
            nodes = gmsh.model.mesh.getElementsByType(tetrahedron, entity)[1].reshape(-1, len(from_gmsh))
            elements.append(nodes[:, from_gmsh])
            materials.append(np.full(len(nodes), group))
    elements, materials = np.vstack(elements), np.concatenate(materials)

    # Nodes numbered from 1 in tag order. Nodes outside every element would carry no stiffness.
    used = np.unique(elements)
    index = np.zeros(used.max() + 1, dtype=int)
    index[used] = np.arange(1, len(used) + 1)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    position = np.empty((len(used), 3))
    keep = np.isin(tags, used)
    position[index[tags[keep]] - 1] = coords.reshape(-1, 3)[keep]

    fixed = np.zeros(len(used), dtype=int)
    for group, mask in fixations.items():
        fixed[index[gmsh.model.mesh.getNodesForPhysicalGroup(2, group)[0]] - 1] |= mask

    opposite = {}                      # vertices opposite each tetrahedron face, keyed by the sorted face vertices
    for e in elements[:, :4]:
        for k in range(4):
            opposite.setdefault(tuple(sorted(np.delete(e, k))), []).append(e[k])
    faces = []
    for group in surfaces:
        for entity in gmsh.model.getEntitiesForPhysicalGroup(2, group):
            for f in gmsh.model.mesh.getElementsByType(triangle, entity)[1].reshape(-1, len(reversed_face)):
                inside = opposite[tuple(sorted(f[:3]))]
                assert len(inside) == 1, f"surface group {group} has a face inside the volume, with no outward side"
                a, b, c, d = (position[index[n] - 1] for n in (*f[:3], inside[0]))
                if np.cross(b - a, c - a) @ (d - a) > 0:
                    f = f[reversed_face]
                faces.append((f, group))

    with open(directory / f"{stem}.node", "w") as out:
        out.write(f"{len(used)} 3 1 0\n")
        for n, (x, y, z) in enumerate(position):
            out.write(f"{n + 1} {x:.17g} {y:.17g} {z:.17g} {fixed[n]}\n")
    with open(directory / f"{stem}.ele", "w") as out:
        out.write(f"{len(elements)} {len(from_gmsh)} 1\n")
        for n, (e, m) in enumerate(zip(index[elements], materials), 1):
            out.write(f"{n} {' '.join(map(str, e))} {m}\n")
    with open(directory / f"{stem}.sur", "w") as out:
        out.write(f"{len(faces)} {len(reversed_face)} 2\n")
        for n, (f, group) in enumerate(faces, 1):
            out.write(f"{n} {' '.join(map(str, index[f]))} {group} {group}\n")


"""Second-order tetrahedral meshes from gmsh, in the tetgen files and node order of CardioMechanics.

gmsh places the mid-edge nodes of a second-order mesh on the model geometry, so a curved boundary
is represented to the order of the element instead of by its chords. The T10 basis functions at gmsh's
quadrature points, in the same node order, integrate over such meshes. Requires gmsh.
"""
import numpy as np

TETRAHEDRON4, TETRAHEDRON10, TRIANGLE6 = 4, 11, 9        # gmsh element types
# CardioMechanics T10 local nodes 9 and 10 sit on the edges (2,4) and (3,4); gmsh numbers them the
# other way round. Vertices, the other mid-edge nodes and six-node triangles agree.
T10_FROM_GMSH = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]


def t10_quadrature(rule):
    """gmsh's integration rule on the reference tetrahedron. Returns the weights (q), the T10 shape
    functions (q, 10) and their reference gradients (q, 10, 3) in the CardioMechanics node order, and
    the linear shape functions of the vertices (q, 4), all at the q points."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        points, weights = gmsh.model.mesh.getIntegrationPoints(TETRAHEDRON10, rule)
        N = gmsh.model.mesh.getBasisFunctions(TETRAHEDRON10, points, "Lagrange")[1]
        dN = gmsh.model.mesh.getBasisFunctions(TETRAHEDRON10, points, "GradLagrange")[1]
        L = gmsh.model.mesh.getBasisFunctions(TETRAHEDRON4, points, "Lagrange")[1]
    finally:
        gmsh.finalize()
    q = len(weights)
    return (weights, N.reshape(q, 10)[:, T10_FROM_GMSH], dN.reshape(q, 10, 3)[:, T10_FROM_GMSH],
            L.reshape(q, 4))


def write_tetgen(directory, stem, surfaces, fixations):
    """Write the second-order tetrahedra of the current gmsh model as tetgen stem.node, stem.ele and
    stem.sur into directory.

    Each element carries the tag of its 3D physical group as material. The six-node faces of every 2D
    physical group in surfaces, which must lie on the boundary of the volume, are written with the
    group tag as surface index, ordered so that their normals point out of the volume. fixations maps
    a 2D physical group to the fixed components of its nodes as a bit mask, 1 for x, 2 for y and 4
    for z, combined over the groups a node lies in.
    """
    import gmsh

    elements, materials = [], []
    for _, group in gmsh.model.getPhysicalGroups(3):
        for entity in gmsh.model.getEntitiesForPhysicalGroup(3, group):
            nodes = gmsh.model.mesh.getElementsByType(TETRAHEDRON10, entity)[1].reshape(-1, 10)
            elements.append(nodes[:, T10_FROM_GMSH])
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
            for f in gmsh.model.mesh.getElementsByType(TRIANGLE6, entity)[1].reshape(-1, 6):
                inside = opposite[tuple(sorted(f[:3]))]
                assert len(inside) == 1, f"surface group {group} has a face inside the volume, with no outward side"
                a, b, c, d = (position[index[n] - 1] for n in (*f[:3], inside[0]))
                if np.cross(b - a, c - a) @ (d - a) > 0:
                    f = f[[0, 2, 1, 5, 4, 3]]
                faces.append((f, group))

    with open(directory / f"{stem}.node", "w") as out:
        out.write(f"{len(used)} 3 1 0\n")
        for n, (x, y, z) in enumerate(position):
            out.write(f"{n + 1} {x:.17g} {y:.17g} {z:.17g} {fixed[n]}\n")
    with open(directory / f"{stem}.ele", "w") as out:
        out.write(f"{len(elements)} 10 1\n")
        for n, (e, m) in enumerate(zip(index[elements], materials), 1):
            out.write(f"{n} {' '.join(map(str, e))} {m}\n")
    with open(directory / f"{stem}.sur", "w") as out:
        out.write(f"{len(faces)} 6 2\n")
        for n, (f, group) in enumerate(faces, 1):
            out.write(f"{n} {' '.join(map(str, index[f]))} {group} {group}\n")


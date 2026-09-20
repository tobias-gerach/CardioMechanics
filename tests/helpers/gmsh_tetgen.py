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


def closed_surface(directory, stem, surface):
    """Whether the faces of surface index `surface` in the tetgen files enclose a volume that does not
    depend on the point the tetrahedra spanning it are taken from, which holds exactly when the faces
    close. This is the criterion of CBCirculationCavity::ClosedSurfaceCheck, applied to the files the
    solver reads, in its units: coordinates in metres, volume in millilitres."""
    nodes = 1e-3 * np.loadtxt(directory / f"{stem}.node", skiprows=1, usecols=(1, 2, 3))
    rows = np.loadtxt(directory / f"{stem}.sur", skiprows=1, dtype=int)
    faces = rows[rows[:, -1] == surface][:, 1:-2] - 1
    # The solver spans a face by a fan from its first node, over the polygon through all of its nodes:
    # the triangle itself for a three-node face, the hexagon of vertices and mid-edge nodes for a six.
    polygon = nodes[faces][:, [0, 3, 1, 4, 2, 5] if faces.shape[1] == 6 else [0, 1, 2]]
    a, b, c = polygon[:, :1], polygon[:, 1:-1], polygon[:, 2:]
    areas = np.cross(b - a, c - a)

    def volume(reference):
        return 1e6 * np.einsum("fkj,fkj->", a - reference, areas) / 6

    return abs(volume(np.ones(3)) - volume(np.zeros(3))) <= 1e-10


# The four triangles a six-node face refines into, in its node order, as
# CBModelLoaderTetgen::LoadSurfaces writes them: the three corner triangles and the middle one.
SUB_TRIANGLES = ((0, 3, 5), (3, 1, 4), (3, 4, 5), (5, 4, 2))


def _fan(faces, group, surface, material):
    """Three-node triangles closing the open boundary of the faces of `group`, as a fan from one of
    its boundary nodes, tagged with `material` and `surface`.

    A boundary edge is one that a single face carries. Taking each in the direction opposite to the
    face that carries it orients the fan consistently with that surface, so the two together bound a
    volume. The fan adds no node: every triangle is spanned by nodes already on the boundary, which
    is what keeps them attached to the elements behind that surface. A node of its own would sit in
    no tetrahedron, and so carry neither stiffness nor mass.
    """
    edges = set()
    for f, g, _ in faces:
        if g == group:
            for edge in ((int(f[0]), int(f[1])), (int(f[1]), int(f[2])), (int(f[2]), int(f[0]))):
                if edge[::-1] in edges:
                    edges.remove(edge[::-1])
                else:
                    edges.add(edge)
    boundary = sorted(edges)
    assert boundary, f"surface group {group} has no open boundary to close"
    apex = boundary[0][0]
    return [(np.array([apex, v, u]), material, surface) for u, v in boundary if apex not in (u, v)]


def write_tetgen(directory, stem, surfaces, fixations, inward=(), refine=False, close=None):
    """Write the tetrahedra of the current gmsh model, first- or second-order, as tetgen stem.node,
    stem.ele and stem.sur into directory.

    With refine, each six-node face is written as the four three-node triangles it refines into,
    which load the same six nodes. close maps a material index to the surface group it closes with a
    fan of three-node triangles, which needs refine on a second-order mesh: the fan spans boundary
    nodes alone, and only a refined surface has every node of its boundary on that boundary.

    Each element carries the tag of its 3D physical group as material. surfaces maps a 2D physical
    group to the surface index its triangles carry, or is an iterable of groups, which gives each
    its own tag; the tag is always written as the material index, so groups that share a surface
    index stay distinguishable there. The triangles are of the order of the tetrahedra and are
    ordered so that their normals point out of the volume, or into it for a group listed in inward,
    which is what a cavity surface needs: it bounds the cavity on its other side, and the cavity
    encloses a positive volume only if the normals point out of the cavity and so into the wall. A
    group off the boundary of the volume, such as the surface closing a cavity, has no outward side
    and keeps the orientation gmsh gave it. fixations maps a 2D physical group to the fixed
    components of its nodes as a bit mask, 1 for x, 2 for y and 4 for z, combined over the groups a
    node lies in.
    """
    import gmsh

    if not isinstance(surfaces, dict):
        surfaces = {group: group for group in surfaces}
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

    # Nodes numbered from 1 in tag order. A node of a written surface is kept even when no element
    # holds it; a node in neither carries no stiffness and is dropped.
    surface_nodes = [gmsh.model.mesh.getNodesForPhysicalGroup(2, group)[0] for group in surfaces]
    used = np.unique(np.concatenate([elements.ravel().astype(int)] + [n.astype(int) for n in surface_nodes]))
    index = np.zeros(used.max() + 1, dtype=int)
    index[used] = np.arange(1, len(used) + 1)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    position = np.empty((len(used), 3))
    keep = np.isin(tags, used)
    position[index[tags[keep]] - 1] = coords.reshape(-1, 3)[keep]

    fixed = np.zeros(len(used), dtype=int)
    for group, mask in fixations.items():
        rows = index[gmsh.model.mesh.getNodesForPhysicalGroup(2, group)[0]] - 1
        assert np.all(rows >= 0), f"fixation group {group} has a node in no element and no written surface"
        fixed[rows] |= mask

    opposite = {}                      # vertices opposite each tetrahedron face, keyed by the sorted face vertices
    for e in elements[:, :4]:
        for k in range(4):
            opposite.setdefault(tuple(sorted(np.delete(e, k))), []).append(e[k])
    faces = []
    for group, surface in surfaces.items():
        for entity in gmsh.model.getEntitiesForPhysicalGroup(2, group):
            for f in gmsh.model.mesh.getElementsByType(triangle, entity)[1].reshape(-1, len(reversed_face)):
                inside = opposite.get(tuple(sorted(f[:3])), ())
                assert len(inside) < 2, f"surface group {group} has a face inside the volume, with two outward sides"
                if inside:
                    a, b, c, d = (position[index[n] - 1] for n in (*f[:3], inside[0]))
                    if (np.cross(b - a, c - a) @ (d - a) > 0) != (group in inward):
                        f = f[reversed_face]
                faces.append((f, group, surface))
    if refine:
        assert len(reversed_face) == 6, "there is nothing to refine on a first-order mesh"
        faces = [(f[list(triangle)], group, surface)
                 for f, group, surface in faces for triangle in SUB_TRIANGLES]
    for material, group in (close or {}).items():
        faces += _fan(faces, group, surfaces[group], material)

    with open(directory / f"{stem}.node", "w") as out:
        out.write(f"{len(used)} 3 1 0\n")
        for n, (x, y, z) in enumerate(position):
            out.write(f"{n + 1} {x:.17g} {y:.17g} {z:.17g} {fixed[n]}\n")
    with open(directory / f"{stem}.ele", "w") as out:
        out.write(f"{len(elements)} {len(from_gmsh)} 1\n")
        for n, (e, m) in enumerate(zip(index[elements], materials), 1):
            out.write(f"{n} {' '.join(map(str, e))} {m}\n")
    with open(directory / f"{stem}.sur", "w") as out:
        out.write(f"{len(faces)} {3 if refine else len(reversed_face)} 2\n")
        for n, (f, group, surface) in enumerate(faces, 1):
            out.write(f"{n} {' '.join(map(str, index[f]))} {group} {surface}\n")


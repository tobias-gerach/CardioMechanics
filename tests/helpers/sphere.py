"""Octant of a thick-walled sphere, meshed by gmsh as curved second-order tetrahedra."""
import numpy as np

from helpers.gmsh_tetgen import write_tetgen

INNER, OUTER = 1.0, 2.0                    # radii of the shell
MATERIAL, CAVITY = 30, 101                 # physical tags of the volume and the inner surface
SYMMETRY_PLANES = (11, 12, 13)             # physical tags of the planes X_i = 0, i = 0, 1, 2


def write_sphere_octant(directory, size):
    """Write the octant X, Y, Z >= 0 of the shell INNER <= |X| <= OUTER, meshed by gmsh at element
    size `size`, as tetgen sphere.node, sphere.ele and sphere.sur into directory. Every node on the
    plane X_i = 0 has component i fixed, so the three plane faces are symmetry planes. The inner
    surface is written as six-node faces with surface index CAVITY."""
    import gmsh

    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        shell, _ = occ.cut([(3, occ.addSphere(0, 0, 0, OUTER))], [(3, occ.addSphere(0, 0, 0, INNER))])
        octant, _ = occ.intersect(shell, [(3, occ.addBox(0, 0, 0, OUTER, OUTER, OUTER))])
        occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag for _, tag in octant], MATERIAL)

        spheres = []
        for _, tag in gmsh.model.getEntities(2):
            centre = np.array(occ.getCenterOfMass(2, tag))
            if gmsh.model.getType(2, tag) == "Plane":
                gmsh.model.addPhysicalGroup(2, [tag], SYMMETRY_PLANES[int(np.argmin(np.abs(centre)))])
            else:
                spheres.append((np.linalg.norm(centre), tag))
        gmsh.model.addPhysicalGroup(2, [min(spheres)[1]], CAVITY)

        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)
        write_tetgen(directory, "sphere", [CAVITY], {tag: 1 << i for i, tag in enumerate(SYMMETRY_PLANES)})
    finally:
        gmsh.finalize()

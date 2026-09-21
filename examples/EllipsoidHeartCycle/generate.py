#!/usr/bin/env python3

"""Derive the ellipsoid heart cycle's inputs from the reference recovery example's mesh.

The cycle runs on the same geometry the reference recovery example recovered the unloaded state of,
with three things changed. Its base is no longer clamped but held by a Robin boundary, so the
boundary condition column of the node file is zeroed. Its epicardium is supported by a pericardial
Robin boundary that is full strength at the apex and fades to zero at the base, so every face of the
surface file carries a traction scaling as a third attribute. And its tension is driven by a
per-element activation time, the transmural delay from the endocardium.

The element and fibre basis files are read in place from the reference recovery example, since
nothing about them changes. The files written here are not committed; run this script first.

Run as `python3 generate.py` from this directory.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

EXAMPLE = Path(__file__).resolve().parent
REFERENCE_RECOVERY = EXAMPLE.parent / "ReferenceRecovery"
sys.path.insert(0, str(EXAMPLE.parents[1] / "tools" / "python"))

from FitMaterialParameters import cavity_volume  # noqa: E402  (needs the path above)

# The length of one coordinate unit of the mesh in metres. The mesh is in millimetres, and
# LoadUnloadedState reads its node files with this unit hard-coded.
UNIT = 1e-3

# Material indices of the surface file, as helpers.ellipsoid writes them. The endocardium and the
# lid share surface index 1, which is what cavity membership goes by, and are told apart by these.
ENDOCARDIUM, EPICARDIUM = 1, 2

# Transmural conduction velocity in m/s. Activation starts on the endocardium, as it does under
# Purkinje activation, and reaches the epicardium after the delay this velocity gives.
CONDUCTION_VELOCITY = 0.3

# The circulation's arterial and venous volumes at the pressures the cycle starts from, in ml: 1000
# in the arteries, which the arterial compliance puts at 80 mmHg, and 3650 in the veins, which the
# venous compliance puts at the 8 mmHg the unloaded state was recovered at. The ventricle's share is
# the mesh's cavity volume, so the total follows from the mesh.
CIRCULATION_VOLUME_OUTSIDE_THE_VENTRICLE = 4650


def source_files():
    """The reference recovery example's mesh and its recovered unloaded state. Raises if the example
    has not been run: its mesh is generated and its unloaded state is the deliverable of the fit, so
    neither is in the repository."""
    mesh = {suffix: REFERENCE_RECOVERY / "tetgen" / f"ellipsoid.{suffix}"
            for suffix in ("node", "ele", "sur", "bases")}
    unloaded = REFERENCE_RECOVERY / "FitMaterialParameters" / "UnloadedState.node"
    missing = [path for path in list(mesh.values()) + [unloaded] if not path.exists()]
    if missing:
        raise SystemExit(
            "missing " + ", ".join(str(path.relative_to(EXAMPLE.parent)) for path in missing)
            + f"\nRun the reference recovery example first; see {REFERENCE_RECOVERY.name}/README.md."
            " It generates the mesh and fits the material parameters, and its material fit writes"
            " the recovered unloaded state this example starts from.")
    return mesh, unloaded


def check_node_order(mesh_nodes, unloaded, coordinates):
    """The recovered unloaded state has to be in the node order of the mesh file.

    The recovery writes its node file in the mesh's own node order regardless of Mesh.Sorting, so
    row by row it should sit close to the mesh: the recovered state is the mesh deflated, a fraction
    of the ventricle away. A permuted file, from an out-of-date fit or a solver defect, would instead
    pair coordinates with the wrong nodes and land a fraction of the ventricle's own size away.
    """
    recovered = np.loadtxt(unloaded, skiprows=1, usecols=(1, 2, 3))
    assert len(recovered) == len(coordinates), \
        f"{unloaded.name} holds {len(recovered)} nodes against {len(coordinates)} in {mesh_nodes.name}"
    moved = np.linalg.norm(recovered - coordinates, axis=1).max()
    extent = np.linalg.norm(coordinates.max(axis=0) - coordinates.min(axis=0))
    assert moved < 0.5 * extent, (
        f"{unloaded.name} is {moved:.1f} mm from the mesh at its furthest node, over half the"
        f" ventricle's {extent:.1f} mm extent: the two are in different node orders."
        " Re-run the reference recovery example's material fit.")


def read_table(path):
    """The rows of a tetgen file as a whitespace table of strings, without its header line."""
    return [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]


def write_node_file(source, target, in_solid):
    """Copy the node file with its boundary condition column zeroed, which releases the base's
    Dirichlet condition. The coordinates are passed through as text rather than reformatted, so the
    mesh stays the one the unloaded state was recovered on to the last digit.

    Asserts first that every node lies in a tetrahedron. A node outside every one has neither
    element stiffness nor mass, since mass comes from the solid elements, so releasing it would
    leave an empty row in the tangent and the factorization would fail on a zero pivot. The lid is
    the surface that can carry such nodes, and it is a fan over the endocardium's own boundary for
    exactly this reason.
    """
    assert in_solid.all(), (
        f"{(~in_solid).sum()} node(s) of {source.name} lie in no tetrahedron and would be released"
        " with nothing to hold them. Regenerate the reference recovery mesh: its lid has to be a fan"
        " over the endocardium's boundary, not a meshed disc.")
    header, *rows = [line for line in source.read_text().splitlines() if line.strip()]
    released = [line.rsplit(maxsplit=1)[0] + " 0" for line in rows]
    target.write_text("\n".join([header] + released) + "\n")

    fixed = [row[-1] for row in read_table(target) if int(row[-1]) != 0]
    assert not fixed, f"{len(fixed)} node(s) are still fixed"
    assert len(released) == len(rows), f"{len(released)} nodes written against {len(rows)} read"
    return len(released)


def traction_scaling(rows, coordinates):
    """The surface traction scaling of each face: on the epicardium linear in the apico-basal height
    of the face centroid, 1 at the apex and 0 at the base, and 1 on every other face.

    The height is normalized over the epicardial centroids rather than over the mesh, so that the
    apical face reaches 1 and the basal one 0 exactly. Only the epicardial faces are scaled, because
    only the pericardium fades: the base's Robin boundary holds the whole base alike.
    """
    materials = np.array([int(row[-2]) for row in rows])
    centroids = np.array([coordinates[[int(n) - 1 for n in row[:-2]]].mean(axis=0) for row in rows])
    epicardium = materials == EPICARDIUM

    # The long axis of the truncated ellipsoid is z, with the apex at its low end.
    height = centroids[epicardium, 2]
    scaling = np.ones(len(rows))
    scaling[epicardium] = (height.max() - height) / (height.max() - height.min())

    assert np.all((scaling >= 0) & (scaling <= 1)), "a scaling lies outside [0, 1]"
    apex, base = height.argmin(), height.argmax()
    assert scaling[epicardium][apex] == 1 and scaling[epicardium][base] == 0, \
        "the epicardial scaling does not run from 1 at the apex to 0 at the base"
    assert np.all(scaling[~epicardium] == 1), "a face off the epicardium is scaled"
    return scaling


def write_surface_file(source, target, coordinates):
    """Copy the surface file with the traction scaling appended as a third attribute, which is what
    the Robin plugins read. Faces are passed through as text; only the header's attribute count and
    the new column are written."""
    header, *rows = [line for line in source.read_text().splitlines() if line.strip()]
    faces = [line.split()[1:] for line in rows]
    scaling = traction_scaling(faces, coordinates)

    count, nodes, attributes = header.split()
    assert int(attributes) == 2, f"{source.name} already carries {attributes} attributes"
    scaled = [f"{line} {s:.6f}" for line, s in zip(rows, scaling)]
    target.write_text("\n".join([f"{count} {nodes} 3"] + scaled) + "\n")
    return len(scaled)


def activation_times(elements, faces, coordinates):
    """The activation time of every solid element in seconds: the distance from its centroid to the
    nearest endocardial face, at the conduction velocity.

    The lid closes the cavity but is not tissue, so it is no activation source and its faces are
    excluded. Distance is taken to the nodes of the endocardial faces, corner and mid-edge alike,
    which sample the surface at half the element size.
    """
    # A solid element's centroid is the mean of its four corner nodes; the mid-edge nodes of a T10
    # element follow from them and would only weight the corners again.
    centroids = np.array([coordinates[[int(n) - 1 for n in row[:4]]].mean(axis=0) for row in elements])
    endocardium = np.unique([int(n) - 1 for row in faces if int(row[-2]) == ENDOCARDIUM
                             for n in row[:-2]])
    distance = cKDTree(coordinates[endocardium]).query(centroids)[0]
    return UNIT * distance / CONDUCTION_VELOCITY


def write_activation_time_file(target, times):
    """One activation time per solid element, in the order of the element file, which is the order
    the solver indexes its solid elements in. The first line is a header the reader skips."""
    target.write_text("\n".join([f"{len(times)} 1"]
                                + [f"{i} {t:.6f}" for i, t in enumerate(times)]) + "\n")


def main():
    mesh, unloaded = source_files()
    tetgen = EXAMPLE / "tetgen"
    tetgen.mkdir(exist_ok=True)

    coordinates = np.loadtxt(mesh["node"], skiprows=1, usecols=(1, 2, 3))
    check_node_order(mesh["node"], unloaded, coordinates)
    elements = [row[1:] for row in read_table(mesh["ele"])]
    faces = [row[1:] for row in read_table(mesh["sur"])]

    in_solid = np.zeros(len(coordinates), bool)
    in_solid[np.unique([int(n) - 1 for row in elements for n in row[:-1]])] = True
    nodes_written = write_node_file(mesh["node"], tetgen / "ellipsoid.node", in_solid)
    assert nodes_written == len(coordinates), \
        f"{nodes_written} nodes written against {len(coordinates)} in {mesh['node'].name}"
    faces_written = write_surface_file(mesh["sur"], tetgen / "ellipsoid.sur", coordinates)
    assert faces_written == len(faces), \
        f"{faces_written} faces written against {len(faces)} in {mesh['sur'].name}"

    times = activation_times(elements, faces, coordinates)
    assert len(times) == len(elements), \
        f"{len(times)} activation times against {len(elements)} elements"
    # An element next to the endocardium still has its centroid a fraction of an element inside the
    # wall, so the earliest time is a fraction of the transmural delay rather than zero.
    assert times.min() < 0.2 * times.max(), \
        f"the earliest activation, {1e3 * times.min():.1f} ms, is not near the endocardium"
    write_activation_time_file(tetgen / "ellipsoid.lat", times)

    volume = cavity_volume(mesh["node"], mesh["sur"], surface=1, unit=UNIT)
    print(f"{len(coordinates)} nodes, {len(elements)} elements, {len(faces)} faces")
    print(f"activation time  {1e3 * times.min():.1f} to {1e3 * times.max():.1f} ms")
    print(f"cavity volume    {volume:.2f} ml")
    print(f"total volume     {volume + CIRCULATION_VOLUME_OUTSIDE_THE_VENTRICLE:.2f} ml"
          "   (Plugins.Circulation.Circs.Circ_1.InitialConditions.TotalVolume)")
    print(f"unloaded state   {unloaded.relative_to(EXAMPLE.parent)}")


if __name__ == "__main__":
    main()

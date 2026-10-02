"""ConvertT4toT10 adds straight-sided midside nodes to a T4 mesh.

A T10 mesh made this way is compared with its T4 mesh as the same geometry, so the corner nodes
must come out exactly as they went in and each midside node exactly at its edge's midpoint.
"""
import subprocess

import numpy as np

# Coordinates whose shortest round-trip decimal needs all 17 significant digits.
CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * np.pi * 100 + 1 / 3
# The T10 edge order: (0,1) (1,2) (0,2) (0,3) (1,3) (2,3).
EDGES = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]


def test_nodes_round_trip_exactly(tmp_path, binary):
    (tmp_path / "t4.node").write_text("4 3 1 0\n" + "".join(
        f"{i} {x} {y} {z} 0\n" for i, (x, y, z) in enumerate(CORNERS, 1)))
    (tmp_path / "t4.ele").write_text("1 4 1\n1 1 2 3 4 5\n")
    subprocess.run([binary("ConvertT4toT10"), "t4.node", "t4.ele", "t10"], cwd=tmp_path, check=True,
                   capture_output=True)
    nodes = np.loadtxt(tmp_path / "t10.node", skiprows=1)[:, 1:4]
    ele = np.loadtxt(tmp_path / "t10.ele", skiprows=1, dtype=int)
    assert np.array_equal(ele, [1, *range(1, 11), 5])
    assert np.array_equal(nodes[:4], CORNERS)
    assert np.array_equal(nodes[4:], [(CORNERS[a] + CORNERS[b]) / 2 for a, b in EDGES])

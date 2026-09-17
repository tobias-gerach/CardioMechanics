#!/usr/bin/env python3
"""Cost of the direct and the AMG linear solver presets over a refined displacement-only mesh family.

The presets of Solver.LinearSolver differ only in cost, not in the solution they converge to, which
tests/cardiomechanics/test_linear_solver.py already asserts. What a modeller needs to know is where
the iterative path starts to pay: MUMPS factorises in fewer, much more expensive operations than a
GMRES sweep, and its factor grows superlinearly with the mesh, so direct wins on small meshes and
runs out of memory on large ones. This script measures that crossover.

The family is the inflated thick-walled sphere octant of test_sphere_convergence, meshed by gmsh at
SIZES and run as displacement-only T10 under the Static solver, so every level solves the same
problem at a different resolution. T10 alone: it carries the most unknowns per element of the
displacement-only types and is what the ventricle-scale cases here use, and the crossover is a
count of unknowns, which transfers to T4. Each level runs under every preset at every rank count of
RANKS. Reported per run, from the binary's own PETSc output:

  its/solve   Krylov iterations per linear solve, averaged over the Newton steps of the run
              (1 for the direct presets, which are preonly). This is the number a preconditioner is
              judged by: it should stay flat under refinement if the preconditioner is optimal.
  ksp, pc     KSPSolve and PCSetUp time from -log_view, the max over the ranks. For the direct
              presets PCSetUp holds the factorisation, which is where their time goes. Static
              rebuilds its SNES every time step, so for the AMG presets PCSetUp holds one full
              preconditioner setup per step, not one per run.
  wall        wall clock of the whole process: reading the mesh, assembly, solving and export, so
              it is the number a modeller feels. The meshes are written once, before the runs, so
              meshing is not in it.
  mem         peak process memory summed over the ranks, from -memory_view, in MiB.

A run that fails (a Krylov solve that does not converge, or a factorisation that does not fit) is
reported as such, and the finer levels of that preset at that rank count are skipped.

Findings (Apple M-series laptop, 8 cores, PETSc 3.24, KAPPA = 50):

  amg is mesh-independent. GAMG takes 90 to 110 iterations per solve at every level and every rank
  count, over a 33-fold refinement, so the rigid-body near-null space is doing its job and the cost
  of a solve grows with the unknowns alone.

  direct costs what a factorisation costs. Its time is almost all PCSetUp and grows by a factor of
  700 over that refinement, against 33 for the unknowns, and its memory by a factor of 44.

  The crossover in serial wall time is at about 30k unknowns: 14.9 s for direct against 15.1 s for
  amg at 28k. At 83k amg is twice as fast and takes half the memory, at 182k it is 3.6 times as
  fast (123 s against 448 s) and takes 1.8 GB against 5.0 GB. The crossover is where a solve of a
  ventricle mesh starts to be worth running iteratively; memory is the harder limit of the two,
  since direct is the one that stops fitting.

  The crossover sits at the same size on 2 and 4 ranks. Neither preset scales well: from 1 to 4
  ranks at 182k, direct goes 448 s -> 216 s and amg 123 s -> 85 s, while total memory rises for
  both, steeply for direct (5.0 GB -> 7.2 GB). What limits it was not chased down. The fixture
  leaves Solver.DomainDecomposition off, so the ranks get contiguous ranges of the tetgen node
  order rather than a partition; turning it on at 28k unknowns on 4 ranks left the iteration count
  where it was (90.8 against 92.5) and made every run slower, the partitioning step included. On
  one laptop these rank counts say nothing about scaling on a cluster either way.

  amg-hypre converges in a third of the iterations of amg (30 to 43, also flat) and is 3 to 4 times
  slower in wall time at every level, because each iteration and each setup costs far more.
  BoomerAMG runs here without the near-null space, which it uses only under nodal or interpolation
  options; try those through Options before choosing this preset.

  Both AMG presets degrade badly as the material approaches incompressibility, where T10 locks.
  Rerun with KAPPA = 1000, the fixture's own value: at 5.6k unknowns amg then needs 460 iterations
  per solve instead of 97 and amg-hypre 540 instead of 30, which makes both slower than direct at
  that size, where at KAPPA = 50 amg already matches it. The iterative path is for moderate kappa;
  a nearly incompressible model belongs in a mixed element, which keeps the direct solve.

Runtime of the full sweep: about an hour. Run from anywhere; the binary is found as by the tests,
so set CM_BIN_DIR to the Release build.
"""
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3] / "tests"
sys.path[:0] = [str(TESTS)]
from conftest import _find_binary                     # noqa: E402
from helpers.run import run_binary                    # noqa: E402
from helpers.sphere import write_sphere_octant        # noqa: E402

FIXTURE = TESTS / "cardiomechanics" / "fixtures" / "sphere_octant.xml"
SIZES = (0.3, 0.2, 0.15, 0.1, 0.075)    # gmsh element sizes, about 5.6k to 180k displacement unknowns
KAPPA = 50                              # bulk over shear modulus, as in the benchmark2015 cases
PRESETS = ("direct", "amg", "amg-hypre")
RANKS = (1, 2, 4)
TIMEOUT = 1800                          # per run; a direct solve that needs longer has lost anyway

# -log_view rows are "<event> <count> <count ratio> <time> <time ratio> ...", the time already the
# max over the ranks.
EVENT_TIME = r"^{}\s+\d+\s+\S+\s+(\S+)\s"
ITERATIONS = re.compile(r"Linear mech_ solve converged due to \S+ iterations (\d+)")
MEMORY = re.compile(r"Maximum \(over computational time\) process memory:\s+total (\S+)")


def settings(preset):
    """The fixture as displacement-only T10 at KAPPA under preset."""
    text = FIXTURE.read_text()
    for old, new in [("<Type>T10P1</Type>", "<Type>T10</Type>"), ("<k>1000</k>", f"<k>{KAPPA}</k>"),
                     ("<Type>Static</Type>",
                      f"<Type>Static</Type><LinearSolver><Preset>{preset}</Preset></LinearSolver>")]:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    return text


def measure(binary, env, wd, mesh_dir, preset, ranks):
    """Run one case in the empty directory wd off the mesh already written to mesh_dir, and return
    its measurements, or why it did not finish."""
    (wd / "Results").mkdir()
    (wd / FIXTURE.name).write_text(settings(preset).replace("./tetgen/", f"{mesh_dir}/"))
    args = ["-settings", FIXTURE.name, "-mech_ksp_converged_reason", "-log_view", "-memory_view"]
    start = time.perf_counter()
    try:
        # Absolute: the run's own directory is the working directory of the process.
        proc = run_binary(Path(binary("CardioMechanics")).resolve(), args, cwd=wd, env=env,
                          timeout=TIMEOUT, np=ranks if ranks > 1 else None, check=False)
    except subprocess.TimeoutExpired:
        return "timeout"
    wall = time.perf_counter() - start
    out = proc.stdout + proc.stderr
    iterations = [int(n) for n in ITERATIONS.findall(out)]
    if proc.returncode != 0 or "SIMULATION FAILED" in out or not iterations:
        return "failed"
    return {"its/solve": sum(iterations) / len(iterations),
            "ksp [s]": float(re.search(EVENT_TIME.format("KSPSolve"), out, re.M).group(1)),
            "pc [s]": float(re.search(EVENT_TIME.format("PCSetUp"), out, re.M).group(1)),
            "wall [s]": wall,
            "mem [MiB]": float(MEMORY.search(out).group(1)) / 2**20}


def table(ranks, dofs, rows):
    """One rank count: a line per level and preset, the levels in order of refinement. A preset is
    skipped on the levels beyond the one it failed on."""
    columns = ("its/solve", "ksp [s]", "pc [s]", "wall [s]", "mem [MiB]")
    lines = [f"np = {ranks}", f"{'dof':>9}  {'preset':<11}" + "".join(f"{c:>11}" for c in columns)]
    for size in SIZES:
        for preset in PRESETS:
            row = rows.get((size, preset), "skipped")
            values = ("".join(f"{row[c]:11.1f}" for c in columns) if isinstance(row, dict)
                      else f"{row:>11}")
            lines.append(f"{dofs[size]:9d}  {preset:<11}" + values)
    return "\n".join(lines)


if __name__ == "__main__":
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory() as tmp:
        meshes, dofs = {}, {}
        for size in SIZES:
            meshes[size] = Path(tmp) / f"mesh_{size}"
            meshes[size].mkdir()
            write_sphere_octant(meshes[size], size, order=2)
            with open(meshes[size] / "sphere.node") as f:      # the header counts the nodes
                dofs[size] = 3 * int(f.readline().split()[0])
        for ranks in RANKS:
            rows, failed = {}, set()
            for size in SIZES:
                for preset in PRESETS:
                    if preset in failed:
                        continue
                    wd = Path(tmp) / f"{ranks}_{size}_{preset}"
                    wd.mkdir()
                    rows[size, preset] = row = measure(_find_binary, env, wd, meshes[size], preset, ranks)
                    if not isinstance(row, dict):
                        failed.add(preset)
                    print(f"  {ranks} {size} {preset}: {row}", file=sys.stderr, flush=True)
            print(table(ranks, dofs, rows) + "\n", flush=True)

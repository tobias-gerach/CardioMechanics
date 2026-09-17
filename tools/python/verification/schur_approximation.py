#!/usr/bin/env python3
"""Cost of the Schur approximations of the fieldsplit preset over kappa and over a refined mesh family.

The textbook Schur approximation for a mixed element is the A11 block, which the perturbed
constraint of ADR-0002 makes a pressure mass matrix scaled by -1/kappa. That block vanishes as kappa
grows, which is the regime mixed elements exist for, so whether the approximation survives it is a
measurement rather than a guess. This script makes it, against selfp, the approximation
A10 inv(diag(A00)) A01 PETSc assembles itself and the preset selects (ADR-0007), and against the
direct solve the preset replaces.

The problem is the inflated thick-walled sphere octant of test_sphere_convergence, run under the
Static solver as each of the two mixed element types. Both arms are swept twice: over kappa on the
second level of the mesh family, and over the whole family at the fixture's kappa = 1000 at every
rank count of RANKS. The two families are meshed at the element sizes that put them on the same
ladder of displacement unknowns, so a level of one compares with the level of the other. The columns are those of linear_solver.py, whose measure() this shares:

  its/solve   outer Krylov iterations per linear solve, averaged over the Newton steps of the run
              (1 for direct, which is preonly). An approximation of the Schur complement is judged
              by this: a good one keeps it flat under both sweeps.
  ksp, pc     KSPSolve and PCSetUp time from -log_view, the max over the ranks. For direct, PCSetUp
              holds the factorisation, which is where its time goes; for fieldsplit it holds the
              multigrid setup of the u block, which is cheap, and nearly all of its time is in
              KSPSolve.
  wall        wall clock of the whole process, so it is the number a modeller feels.
  mem         peak process memory summed over the ranks, from -memory_view, in MiB.

Findings (Apple M-series laptop, 8 cores, PETSc 3.24):

  a11 fails at the kappa mixed elements are run at. At 13.1k displacement unknowns it takes 79
  iterations per solve at kappa = 10 and 1145 at kappa = 1e6: close to a doubling over each of the
  first three decades, and half as much again over the last. The growth decelerates but does not
  stop, and there is no plateau in sight: the block being approximated is proportional to 1/kappa,
  so the approximation degrades with the very parameter a mixed element is chosen for. The fixture
  runs at kappa = 1000, where it already costs 292.

  selfp is kappa-independent. It takes 52 iterations at kappa = 10 and 62 from kappa = 100 to
  kappa = 1e6, unchanged to the last digit over four decades. That is a factor of 4.7 below a11 at
  the fixture kappa and of 18 at kappa = 1e6. diag(A00) does not know about kappa, so neither does
  the approximation.

  Under refinement the two swap roles. At kappa = 1000 over a 33-fold refinement (5.6k to 182k
  displacement unknowns) a11 is flat at 286 to 308 iterations, so what limits it is kappa alone,
  while selfp grows from 52 to 137, a little slower than the cube root of the unknowns. selfp is
  thus not mesh-independent, and the growth would eventually eat its margin; over the range a
  ventricle mesh spans it keeps a factor of two or more, and the mesh dependence is the milder
  defect of the two, since kappa is a material choice the modeller cannot trade against cost.

  Both beat the direct solve where it matters. Serially at kappa = 1000, selfp reaches parity in
  wall time at 28k unknowns (12.5 s against 12.0 s), is 1.7 times faster at 83k and 2.6 times at
  182k (141 s against 362 s), and takes 2.5 GB there against 5.7 GB; a11 reaches parity only at
  83k. Memory is the limit that matters, since it is the one that stops a ventricle mesh from
  running at all, and both fieldsplit configurations take well under half of the factorisation.
  The picture is the same on 2 and 4 ranks. Memory rises with the rank count for every
  configuration, steeply (selfp at 182k: 2.5 GB, 4.8 GB, 6.8 GB on 1, 2 and 4 ranks), as
  linear_solver.py already found for the displacement-only presets; the fixture leaves
  Solver.DomainDecomposition off.

  T4MINI says the same, with one difference. Its kappa sweep at 11.7k unknowns runs a11 from 67
  iterations at kappa = 10 to 622 at 1e6 and selfp from 46 to 59, flat from kappa = 1000 up, so the
  kappa dependence is the same defect one element type milder. Under refinement, though, a11 is not
  flat here as it is on T10P1: it grows 200, 284, 366, 503 over the first four levels, so on MINI it
  carries a mesh dependence on top of the kappa one and is behind selfp on both counts. selfp grows
  48 to 153 over the same range. At the finest level, 172k unknowns, only selfp finished a serial
  run inside the 1800 s cap, at 1563 s and 6.9 GiB; direct and a11 did not. Wall time on this arm is
  mostly the element kernel rather than the solver, the linear mesh carrying about eight times the
  elements of the T10P1 mesh at equal unknowns: at 11.7k unknowns a direct run spends 9.6 s in
  PCSetUp and 48 s on the clock. The iteration columns are what compares the approximations there.

  So a11 is not the right default and selfp is. The out-of-scope item of the spec, a pressure mass
  matrix assembled specifically as a Schur preconditioner, is not needed: selfp already removes the
  kappa dependence that motivated it, at no assembly cost. Its remaining weakness is the mesh
  dependence, which a pressure mass matrix would not fix either, being the same kind of
  approximation of the same operator.

Runtime: about 50 minutes for the T10P1 arm and 5 hours for the T4MINI one, whose element kernel
and eight times the elements make every run longer. An element type on the command line runs that
arm alone. Run from anywhere; the binary is found as by the
tests, so set CM_BIN_DIR to the Release build.
"""
import os
import sys
import tempfile
from pathlib import Path

TESTS = Path(__file__).resolve().parents[3] / "tests"
sys.path[:0] = [str(Path(__file__).resolve().parent), str(TESTS)]
from conftest import _find_binary                     # noqa: E402
from helpers.sphere import write_sphere_octant        # noqa: E402
from linear_solver import FIXTURE, measure            # noqa: E402

# name -> (preset, Solver.LinearSolver.Options). The options overwrite the preset, so the two rows
# differ in the Schur approximation alone, and stay what they say they are if the preset changes.
CONFIGS = {"direct": ("direct", ""),
           "a11": ("fieldsplit", "-mech_pc_fieldsplit_schur_precondition a11"),
           "selfp": ("fieldsplit", "-mech_pc_fieldsplit_schur_precondition selfp")}
# element type -> (mesh order, gmsh element sizes). A linear mesh carries about an eighth of the
# nodes of a quadratic one at the same element size, so the T4MINI sizes are the ones that put both
# families on the same ladder of displacement unknowns, 5k to 180k, which is what the comparison is
# over. The T10P1 sizes are those of linear_solver.py.
ELEMENTS = {"T10P1": (2, (0.3, 0.2, 0.15, 0.1, 0.075)),
            "T4MINI": (1, (0.14, 0.1, 0.075, 0.048, 0.0374))}
KAPPA_SWEEP = (10, 100, 1000, 10000, 100000, 1000000)
KAPPA = 1000                            # of the mesh family; the fixture's own value
KAPPA_LEVEL = 1                         # mesh of the kappa sweep, the second level of the family
RANKS = (1, 2, 4)
COLUMNS = ("its/solve", "ksp [s]", "pc [s]", "wall [s]", "mem [MiB]")


def settings(config, kappa, element):
    """The fixture as the mixed element type at kappa under config. A linear mesh carries its cavity
    as three-node faces, where the quadratic one of the shipped fixture carries six-node ones."""
    preset, options = CONFIGS[config]
    order, _ = ELEMENTS[element]
    text = FIXTURE.read_text()
    for old, new in [("<k>1000</k>", f"<k>{kappa}</k>"),
                     ("<Type>T10P1</Type>", f"<Type>{element}</Type>"),
                     ("<Type>T6</Type>", f"<Type>{'T6' if order == 2 else 'T3'}</Type>"),
                     ("<Type>Static</Type>",
                      f"<Type>Static</Type><LinearSolver><Preset>{preset}</Preset>"
                      f"<Options>{options}</Options></LinearSolver>")]:
        assert text.count(old) == 1, f"{FIXTURE.name}: cannot substitute {old}"
        text = text.replace(old, new)
    return text


def table(title, label, entries, rows):
    """One table: a line per entry (text, key) and config, a run that did not finish in place of its
    measurements."""
    lines = [title, f"{label:>9}  {'config':<8}" + "".join(f"{c:>11}" for c in COLUMNS)]
    for text, key in entries:
        for config in CONFIGS:
            row = rows.get((key, config), "skipped")
            values = ("".join(f"{row[c]:11.1f}" for c in COLUMNS) if isinstance(row, dict)
                      else f"{row:>11}")
            lines.append(f"{text:>9}  {config:<8}" + values)
    return "\n".join(lines)


def sweep(env, tmp, name, element, cases, ranks):
    """Run every case (key, mesh, kappa) under every config, skipping the finer cases of a config
    once it has failed, and return the rows."""
    rows, failed = {}, set()
    for key, mesh, kappa in cases:
        for config in CONFIGS:
            if config in failed:
                continue
            wd = Path(tmp) / f"{element}_{name}_{ranks}_{key}_{config}"
            wd.mkdir()
            rows[key, config] = row = measure(_find_binary, env, wd, mesh,
                                              settings(config, kappa, element), ranks)
            if not isinstance(row, dict):
                failed.add(config)
            print(f"  {element} {name} np={ranks} {key} {config}: {row}", file=sys.stderr, flush=True)
    return rows


if __name__ == "__main__":
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    # An element type on the command line runs that arm alone, so a family can be remeasured without
    # the other one, which takes as long again.
    elements = sys.argv[1:] or list(ELEMENTS)
    with tempfile.TemporaryDirectory() as tmp:
        for element in elements:
            order, sizes = ELEMENTS[element]
            meshes, dofs = {}, {}
            for size in sizes:
                meshes[size] = Path(tmp) / f"mesh_{element}_{size}"
                meshes[size].mkdir()
                write_sphere_octant(meshes[size], size, order=order)
                with open(meshes[size] / "sphere.node") as f:  # the header counts the nodes
                    dofs[size] = 3 * int(f.readline().split()[0])

            kappa_size = sizes[KAPPA_LEVEL]
            cases = [(kappa, meshes[kappa_size], kappa) for kappa in KAPPA_SWEEP]
            rows = sweep(env, tmp, "kappa", element, cases, 1)
            print(table(f"{element}, np = 1, {dofs[kappa_size]} displacement unknowns", "kappa",
                        [(str(kappa), kappa) for kappa in KAPPA_SWEEP], rows) + "\n", flush=True)

            for ranks in RANKS:
                rows = sweep(env, tmp, "mesh", element,
                             [(size, meshes[size], KAPPA) for size in sizes], ranks)
                print(table(f"{element}, np = {ranks}, kappa = {KAPPA}", "dof",
                            [(str(dofs[size]), size) for size in sizes], rows) + "\n", flush=True)

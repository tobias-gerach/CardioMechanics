# Regression tests

Tests of the CardioMechanics binaries. Each test runs a binary on a fixed input
and compares its output, with numerical tolerances (never byte-exact), to an
independent reference: a closed-form solution, a serial run, PETSc's finite
differences, or a committed golden file.

Coverage:
- **CellModelTest** — all runnable single-cell ionic models (auto-discovered),
  plus coupled ionic+tension runs for Land17. Serial.
- **CardioMechanics** — benchmark2015 Problem1 (Static solver) at `mpirun -np 4`:
  the `Pressure.dat` time series and the final deformed geometry (last VTU point
  coordinates, via meshio). Marked `mpi`.
- **CardioMechanics inverse** — active-stress estimator round trip on the
  ellipsoid example: a forward run with known active tension, then the estimator
  recovering it from the deformed surface; compares the recovered active stress and
  deformation.
- **CardioMechanics dynamic** — mechanics-only dynamic fixture: benchmark2015
  Problem 3 (T10 ellipsoid, pressure and active-tension ramps) run with the
  `NewmarkBeta` solver and Rayleigh damping instead of the benchmark's `Static`
  solver, at `mpirun -np 4`; compares the final deformed geometry. Isolates the
  time integrator from the coupled EM chain. Marked `mpi slow`. The same module
  also runs a creep fixture under both dynamic solvers: stiffness-proportional
  Rayleigh damping only, large enough to overdamp every mode, so the cavity
  volume has to relax with time constant `Beta`; checks that the
  `GeneralizedAlpha` Jacobian fits its preallocation under consistent and lumped
  mass; and, serially and not slow, that a missing or out-of-range `RhoInf` and
  a damping `Type` other than `Rayleigh` are refused.
- **CardioMechanics free vibration** — a cantilever ringing freely after a
  pressure pulse, T10 under `NewmarkBeta` and T10 and P2P1 under
  `GeneralizedAlpha` at five `RhoInf`: the period and logarithmic decrement of
  the first bending mode against the closed form of each scheme, and
  second-order convergence of the displacement.
  The only check of the period and numerical damping of the integrators. Marked
  `slow`.
- **CardioMechanics mixed elements** (`test_p2p1`, `test_t4mini`) — P2P1
  (Taylor-Hood, element type `T10P1`) and T4MINI on a small cantilever: input refusals, static and
  `GeneralizedAlpha` runs, serial against `mpirun` runs, and for P2P1 the
  pressure export and constraint, the Jacobian against PETSc finite differences,
  agreement with T10 at small kappa, and that a run whose time stepping gives up exits
  non-zero. The static parallel test of each element is not marked `slow`; the
  `GeneralizedAlpha` and Robin-boundary parallel tests are marked `mpi slow`.
- **CardioMechanics uniaxial patch** — homogeneous uniaxial tension of a
  distorted box under NeoHooke, T4, T10 and P2P1 (default rule, and the 14-point
  rule for P2P1): nodes against the closed-form stretches, and the P2P1 pressure
  against `kappa (J - 1)`.
- **CardioMechanics rigid rotation** — a distorted box rotated by 90° through
  the points-control plugin under NeoHooke, T4, T10 and P2P1: nodes follow the
  rotation, strain, stress and P2P1 pressure vanish.
- **CardioMechanics sphere convergence** — inflated thick-walled sphere octant
  against its exact radial solution: P2P1 and T4MINI displacement and pressure
  errors converge at the rate of the pairing on the coarse mesh family. Marked
  `slow`.
- **BidomainMatrixGenerator** — assembles the EM01 mono-domain matrices (serial)
  and compares structural/numeric invariants (dims, nnz, Frobenius norm, sums)
  of the stiffness/mass matrices and material vector, read directly from the
  PETSc binary format (no petsc4py needed).
- **acCELLerate** — standalone EP solve on the EM01 cube at `mpirun -np 4`;
  compares the P8 sensor traces (Vm, Cai). Marked `mpi slow`.
- **CardioMechanics EM01** — full electromechanics (`NewmarkBeta` solver +
  acCELLerate plugin + Land17) at `mpirun -np 4`; compares final deformation and
  the coupled P8 sensor traces, and runs the same case on T4MINI under
  `GeneralizedAlpha`. Marked `mpi slow`.
- **ConvertT4toT10**: one tetrahedron converted to T10; the corner nodes read
  back exactly and every midside node sits exactly at its edge's midpoint.

The three EM01 tests share one staged tree and a single matrix-assembly step
(the `em01_root` fixture). The EM01 electromechanics runs are shortened to
`EM01_SIM_LENGTH` (0.05 s) — the full 1 s beat is ~30+ min; 0.05 s still captures
the wavefront reaching the far sensor P8 (~42 ms) and the first clear mechanical
deformation (~30 ms).

Material-law correctness (energy against the published strain energy functions,
PK2 stress as its derivative) is checked below the binary, by the GoogleTests of
`mechanics/tests/ConstitutiveModelTest.cpp` (`ctest --test-dir _build/test`).

## Study scripts

Studies of a formulation print their results and assert nothing, so they live
outside pytest in `tools/python/verification/`. Rerun one after changing what it
studies and compare its output with the numbers in its docstring, or, for the
reference energies, with the values the GoogleTest hard-codes:

- `kappa_sensitivity.py` — strain spread over kappa, T10 against P2P1 and T4
  against T4MINI.
- `inf_sup.py` — numerical inf-sup constant of P2P1 on cube and thin-shell mesh
  families.
- `sphere_convergence.py` — sphere errors and rates over the fine family and both
  quadrature rules, and the T10 rates.
- `law_energies.py` — the reference energies hard-coded in the
  law-level GoogleTest.
- `linear_solver.py` — Krylov iterations, solve time and peak memory of the
  `direct`, `amg` and `amg-hypre` presets over a refined displacement-only mesh
  family at several rank counts; the mesh size from which the iterative presets
  win is read off its tables.
- `schur_approximation.py` — outer iterations, solve time and peak memory of
  `direct` and of the `fieldsplit` preset under the `a11` and the `selfp` Schur
  approximation, over a kappa sweep and over a refined mesh family of each mixed
  element type at several rank counts; which Schur approximation belongs in the preset is read
  off its tables. Rerun it after changing the `fieldsplit` preset, the Schur
  approximation it selects, or the perturbed constraint of ADR-0002 that leaves
  the pressure mass matrix in the `A11` block.

Scripts that run a binary find it as the tests do, so set `CM_BIN_DIR` to the
Release build.

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r tests/requirements.txt
```

## Run

```sh
pytest tests/                 # all
pytest tests/ -k cellmodel    # one binary
pytest -m "not slow"          # skip long integration cases
pytest -m "not mpi"           # skip parallel cases
```

## Regenerate goldens

After an *intended* behaviour change, rebaseline and review the diff before
committing:

```sh
pytest tests/ --update-golden
git diff tests/**/golden
```

## Notes

- **Binary discovery:** set `$CM_BIN_DIR` to pin a build directory, otherwise the
  newest match of `_build/*/bin/**/<name>` is used (release/debug/installed
  agnostic).
- **No `kaRootDir` needed:** the binaries resolve their bundled
  `electrophysiology/data` files via `CM_SOURCE_DIR`, baked in at build time, so the
  fixtures set no data-path variable. (`kaRootDir` still overrides it if set.)
- **Goldens** were generated on this machine (PETSc 3.24, serial). Cross-environment
  drift is absorbed by tolerances, not per-platform goldens.

## Known non-functional models

The following cell models do not run standalone in the current repo and are
excluded from the CellModelTest suite (`KNOWN_BROKEN` in
`cellmodel/test_cellmodel.py`). These are pre-existing legacy defects, not
regressions — to be investigated separately:

- `HimenoEtAl_Endo`, `HimenoEtAl_Epi`, `HimenoEtAl_Mid` — each tries to open a
  base `HimenoEtAl.ev` that is not shipped in `electrophysiology/data/`.
- `Kurata` — fails parameter initialization ("Init elphy parameters failed").

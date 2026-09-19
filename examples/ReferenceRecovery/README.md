# Reference configuration recovery with a Klotz-based material fit

A geometry segmented from images is a loaded configuration: it carries the end-diastolic pressure
and the stress that balances it. The `ReferenceRecovery` plugin runs the Sellier fixed-point
iteration with the Rausch augmentation and returns the unloaded reference configuration. This
example turns that plugin on for a truncated-ellipsoid ventricle, and wraps it in the outer loop
that fits the passive material parameters so the simulated end-diastolic pressure-volume relation
matches the empirical Klotz one:

```
mesh  ->  CardioMechanics + ReferenceRecovery  ->  fit Usyk to Klotz  ->  repeat
            (unloaded configuration)               (new parameters)
```

The geometry is the Land et al. (2015) benchmark ellipsoid at refinement level 1, scaled by 3.65 to
a cavity of 120.7 ml, with a lid closing the endocardium so the enclosed volume is well defined.
P2P1 elements, the Usyk law, the base clamped in all directions, and a target pressure of 8 mmHg.

## Cost

About forty minutes on four ranks: five outer iterations of 5 to 9 minutes each. One outer
iteration is one complete recovery, up to six inflations of the ellipsoid under the cycle cap of
five, so the cost per iteration tracks the number of cycles the inner loop needs.

That number is set by the plugin's `Tolerance`, the infinity norm of the nodal residual the recovery
stops at. The settings file puts it at 3e-4 m, where it serves the loop: a recovery at 1e-4 m takes
up to six cycles where this one takes four, and ends at an unloaded volume 0.08 ml away, an eighth
of the loop's volume tolerance. The same run at 1e-4 m took 57 minutes and arrived at the same
parameters. The tolerance is also the accuracy of the deliverable, 0.3 mm on a ventricle 91 mm
long, and the recovered configuration inflates to within 0.3 ml of the target volume rather than
the 0.02 ml of a recovery at 1e-4 m. A reader who needs it tighter lowers `Tolerance` in the last
iteration's settings file and runs that file once more, about 14 minutes at 1e-4 m:

```sh
mpirun -np 4 CardioMechanics -settings FitMaterialParameters/iteration_05/ReferenceRecovery.xml
```

The run overwrites that iteration's records, and the tighter configuration is
`iteration_05/ReferenceRecovery/UnloadedState_Incr1.node`; the work directory's
`UnloadedState.node` stays the loop's.

The loop's own stopping rules are command-line options, since a different geometry or measurement
is a different judgement: `--volume-tolerance`, the unloaded-volume residual it converges at as a
fraction of the measured volume (default 0.005, 0.60 ml here); `--stagnation-tolerance`, the
largest parameter move at which it has stagnated (default 0.01); and `--scaling-bounds`, the
interval each iteration's scalings are clamped to (default 0.2 to 5). The example converges on the
volume and does not reach the stagnation ending. Near the fixed point the parameter move roughly
halves with each iteration; on a run the volume tolerance could not end, a threshold of 0.01 rather
than 0.001 would have saved three iterations that moved the unloaded volume by 0.03 ml.

## Run

```sh
# 1. the mesh, written into tetgen/ (needs gmsh; run from tests/, not from here)
(cd ../../tests && python3 -m helpers.ellipsoid ../examples/ReferenceRecovery/tetgen \
     --level 1 --lid --scale 3.65)

# 2. the loop: recover, fit, rescale, repeat
python3 ../../tools/python/FitMaterialParameters.py ReferenceRecovery.xml \
    --pressure 8 --ranks 4
```

`--pressure` is the measured end-diastolic pressure in mmHg. The measured volume defaults to the
cavity volume computed from the mesh the settings file names, which is what makes the second command
the whole of the workflow. `ReferenceRecovery.xml` itself is never written: each iteration gets its
own copy, carrying that iteration's parameters, under the work directory.

To run the recovery once without fitting anything, use the settings file directly:

```sh
mpirun -np 4 CardioMechanics -settings ReferenceRecovery.xml
```

## Output

Everything lands in the work directory, `FitMaterialParameters/` by default:

- `UnloadedState.node`: the recovered unloaded configuration. This is the deliverable - it is what
  the `LoadUnloadedState` plugin consumes.
- `iterations.csv`: one row per outer iteration, with the material parameters that row's run used,
  the scalings applied to the row below, the unloaded and end-diastolic volumes, both residuals, the
  inner cycle count, the wall time and the number of corrupt-element reports the run printed.
- `PressureVolume.png`: the Klotz relation, its exponential fit, and each iteration's simulated
  pressure-volume curve.
- `iteration_NN/`: each run's settings file, log, plugin records and exported VTUs, and
  `CardioMechanics.out`, everything the run printed.

The solver's output goes to `CardioMechanics.out` rather than to the terminal, which carries the
loop's own lines and a single progress line of the run in flight.

A non-zero corrupt-element count, printed with each iteration's summary, is the Newton line search
meeting an inverted element, `det F <= 0` at a quadrature point, and cutting the step back. The run
recovers from it and the iteration's numbers stand. It says the fit asked for more than the geometry
takes: parameters scaled far enough that a full Newton step inverts elements. The reports themselves,
one per element, rank and evaluation, are in `CardioMechanics.out` and in no log.

The plugin's own records are under `iteration_NN/ReferenceRecovery/`: `PressureVolumeInfo.dat` holds
the pressure and volume of every step, and `CycleInfo.dat` summarizes each fixed-point cycle, whose
residual norm you can watch fall across the cycles of a single run.

## What the authors observed

**The loop converged in five outer iterations.** The unloaded volume ends at 66.86 ml against the
66.65 ml the Klotz relation predicts from the measured pair: a residual of 0.21 ml, where convergence
asks for 0.60, half a percent of the end-diastolic volume. The parameters of that last run are

| | a [Pa] | bff | bss | bnn | bfs | bfn | bns |
|---|---|---|---|---|---|---|---|
| Usyk et al. (2002) | 880 | 8 | 6 | 3 | 12 | 3 | 3 |
| fitted | 156 | 17.1 | 12.8 | 6.4 | 25.7 | 6.4 | 6.4 |

a sixth of the stiffness level at a little over twice the exponents, with the law's anisotropy ratios
preserved, since one factor scales the stiffness and one scales every exponent.

| iteration | cycles | unloaded [ml] | residual [ml] | stiffness scaling | exponent scaling |
|---|---|---|---|---|---|
| 1 | 4 | 73.16 | 6.51 | 0.200 (clamped) | 4.713 |
| 2 | 3 | 86.01 | 19.35 | 0.573 | 0.636 |
| 3 | 4 | 70.56 | 3.91 | 1.867 | 0.698 |
| 4 | 4 | 68.46 | 1.81 | 0.829 | 1.023 |
| 5 | 4 | 66.86 | 0.21 | - | - |

The first step asks for a stiffness level five times lower than Usyk's and is clamped; together with
the near fivefold rise of the exponents it overshoots, and the iterations after it close in on the
Klotz prediction from alternating sides. The end-diastolic volume is within 0.3 ml of the target in
every iteration, so the recovery lands on the target geometry and the fit sees the curve the material
produces.

Convergence is on the unloaded volume, not on the parameters: the last iteration still asked for a
12 % change of the stiffness level. A tighter volume tolerance moves the parameters further; the
update rule's own fixed point, where the simulated relation has the shape of the Klotz one, is the
loop's other ending, stagnation, which also exits zero and reports the unloaded-volume residual it
leaves.

## Notes

The loop's convergence criterion is the unloaded-volume residual alone. The end-diastolic residual is
near-tautological, since the recovery drives the loaded configuration onto the target, and is
reported as a check on the inner loop.

The endocardium and the lid are declared as 6-node surfaces, **not** as the `CAVITY` type. `CAVITY`
is registered against a 3-node triangle, and forcing flat 3-node faces on a quadratic mesh would
distribute the pressure load wrongly: the consistent nodal load of a uniformly loaded 6-node face
sits on the mid-edge nodes, not the vertices. No type declaration is needed, because a 6-node
surface element already derives from the cavity element and the plugin selects cavity elements by
element type. The plugin's error message suggesting otherwise is stale.

The benchmark ellipsoid is scaled rather than used at its original 2.5 ml because the Klotz relation
is empirical and was established on ventricles of physiological size. Its unloaded-volume term is
scale-free, so a fit on a 2.5 ml chamber would run, but presenting one would invite the reader to
take it more seriously than it deserves.

## Sources

- Sellier, M. (2011). An iterative method for the inverse elasto-static problem. *Journal of Fluids
  and Structures* 27(8), 1461-1470. The fixed-point iteration the plugin implements.
- Rausch, M. K., Genet, M., Humphrey, J. D. (2017). An augmented iterative method for identifying a
  stress-free reference configuration in image-based biomechanical modeling. *Journal of
  Biomechanics* 58, 227-231. The augmentation that accelerates it, enabled here.
- Klotz, S. et al. (2006). Single-beat estimation of end-diastolic pressure-volume relationship: a
  novel method with potential for noninvasive application. *American Journal of Physiology - Heart
  and Circulatory Physiology* 291(1), H403-H412. The empirical end-diastolic pressure-volume
  relation the parameters are fitted against.
- Usyk, T. P., LeGrice, I. J., McCulloch, A. D. (2002). Computational model of three-dimensional
  cardiac electromechanics. *Computing and Visualization in Science* 4(4), 249-257. The passive
  material law and its parameters.
- Land, S. et al. (2015). Verification of cardiac mechanics software: benchmark problems and
  solutions for testing active and passive material behaviour. *Proceedings of the Royal Society A*
  471, 20150641. The geometry and its analytic fibre field.

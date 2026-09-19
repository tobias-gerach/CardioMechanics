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

About 20 minutes per outer iteration on four ranks, and 12 iterations to reach its ending, so
roughly 4 hours. One outer iteration is one complete recovery, which is up to six inflations of the
ellipsoid under the cycle cap of five, so the cost per iteration tracks the number of cycles the
inner loop needs and ranges from 7 to 25 minutes. Pass `--iterations 12`: the default cap of 10 stops this example one
iteration short of its own ending, and reaching the cap is an error exit.

## Run

```sh
# 1. the mesh, written into tetgen/ (needs gmsh; run from tests/, not from here)
(cd ../../tests && python3 -m helpers.ellipsoid ../examples/ReferenceRecovery/tetgen \
     --level 1 --lid --scale 3.65)

# 2. the loop: recover, fit, rescale, repeat
python3 ../../tools/python/FitMaterialParameters.py ReferenceRecovery.xml \
    --pressure 8 --ranks 4 --iterations 12
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
  inner cycle count and the wall time.
- `PressureVolume.png`: the Klotz relation, its exponential fit, and each iteration's simulated
  pressure-volume curve.
- `iteration_NN/`: each run's settings file, log, plugin records and exported VTUs.

The plugin's own records are under `iteration_NN/ReferenceRecovery/`: `PressureVolumeInfo.dat` holds
the pressure and volume of every step, and `CycleInfo.dat` summarizes each fixed-point cycle, whose
residual norm you can watch fall across the cycles of a single run.

## What the authors observed

**The loop terminated on parameter stagnation, not on volume.** It took 12 outer iterations, the
last two of them resumed from the tenth's parameters after the default cap stopped the first run,
and the twelfth asked for a parameter change of 0.08 %, under the tenth of a percent below which the
loop calls the fit stagnant. The
parameters settle on

| | a [Pa] | bff | bss | bnn | bfs | bfn | bns |
|---|---|---|---|---|---|---|---|
| Usyk et al. (2002) | 880 | 8 | 6 | 3 | 12 | 3 | 3 |
| fitted | 171 | 16.3 | 12.2 | 6.1 | 24.5 | 6.1 | 6.1 |

a fifth of the stiffness level at twice the exponents, with the law's anisotropy ratios preserved,
since one factor scales the stiffness and one scales every exponent.

The unloaded volume it converges to is 68.0 ml against the 66.65 ml the Klotz relation predicts
from the measured pair: a residual of 1.36 ml, where convergence asks for 0.60, half a percent of
the end-diastolic volume. That residual stopped moving at iteration 5 and did not fall again over
the seven iterations that followed.

Two things hold it there, and the run measures both.

**The fit is at its own fixed point.** Fitting the exponential model to the last iteration's curve
and to the Klotz relation gives

    Klotz      V0 = 66.65 ml   prefactor = 1.0077 mmHg   exponent = 2.7013
    simulated  V0 = 68.02 ml   prefactor = 1.0095 mmHg   exponent = 2.6997

The two relations have the same shape to 0.2 %; the simulated one is the empirical one scaled up by
about 2 % in volume. The model measures a curve's dilation from its own unloaded volume, so its
coefficients describe shape and not size, and once the shapes agree both scalings are 1 however far
apart the two unloaded volumes are. The update rule has no term that closes an absolute offset, so
this is where any run of it ends.

**Most of the offset is the recovery's own pressure bias.** The simulated curve ends at 122.84 ml
where the target is 120.747, the one-step lag described under Notes below. A curve pinned 1.7 % too
large at the loaded end and matched in shape sits about 1.2 % too large at the unloaded end, which
is 1.15 ml of the 1.36 ml residual. The rest is within the scatter of the fits. The bias is
proportional to `Solver.TimeStep / InflationDuration`, so a tenfold smaller time step would remove
most of what separates this run from its convergence criterion, at ten times the cost per
iteration.

A base clamped in all directions carries load and limits how far the cavity can shrink, which is
the reason to expect a residual of this kind on this geometry. On the numbers above it is not the
leading term; the pressure bias is.

Stagnation is an outcome, not a failure: the loop exits zero and the recovered unloaded
configuration it leaves is the deliverable either way.

## Notes

The exported node file and the residual are one solver step behind the volume the plugin reports:
the residual is computed on the configuration at the *start* of the step, while the volume is read
off the freshly displaced one. In every cycle after the first the ramp's last step is the finishing
one, so the recovered reference is the one whose 90 %-of-target-pressure configuration matches the
target - a 10 % pressure bias, exactly one step of the ramp, `Solver.TimeStep / InflationDuration`.
It shrinks with the time step. The end-diastolic volume of the last inflation therefore overshoots
the target by about that much, which is why the loop's convergence criterion is the unloaded-volume
residual alone and the end-diastolic residual is reported as a sanity check on the inner loop.

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

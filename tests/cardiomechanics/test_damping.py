"""High-frequency damping evidence for the generalized-alpha integrator.

Lowering RhoInf is supposed to suppress high-frequency oscillation while leaving
the resolved, low-frequency response alone. That claim is what this module
measures; the RhoInf=1 equivalence in test_dynamic.py only shows the integrator
agrees with Newmark-beta at the non-dissipative end of the family.

The fixture is the mechanics-only ellipsoid of test_dynamic.py, stripped further:
no Rayleigh damping, so the only dissipation left is numerical; no active
tension; and an endocardial pressure switched on at t=0 and then held exactly
constant. Holding it constant is what makes the sweep comparable, because
generalized-alpha evaluates loads at t_n+1 - alphaF*dt and alphaF is a function
of RhoInf: any time-varying load would hand each sweep point a different loading
history and confound the measurement with it.

The 200 Pa step is well below what the fixture can take. Undamped it diverges at
around 800 Pa, a scheme that removes no energy being unable to carry the
high-frequency content a larger step puts in, so RhoInf=1 is the binding case.

The signal is the cavity volume, which ApplyPressureFromFunction writes every
time step. The unused band between LF_MAX and HF_MIN separates the structure's
own ringing -- a ~100 Hz breathing mode and its harmonics, resolved at this step
size and so expected to survive the sweep -- from the unresolved modes carrying
omega*dt >~ 1, which are the ones the scheme is meant to annihilate.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from helpers.compare import read_table
from helpers.run import run_binary

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "examples" / "benchmark2015"
FIXTURE = Path(__file__).parent / "fixtures" / "dynamic_ellipsoid_ringdown.xml"
DEFAULT_RHO_INF = "<RhoInf>1.0</RhoInf>"       # the element _stage rewrites per sweep point

NP = 4                     # CardioMechanics is tested only in parallel, as in test_dynamic
DT = 1e-4                  # Solver.TimeStep in the fixture; also the trace sample interval
STOP_TIME = 0.02           # Solver.StopTime in the fixture
EXPECTED_SAMPLES = int(round(STOP_TIME / DT)) + 1      # t=0 included

# Spanning the full valid range, from the non-dissipative end to maximum damping.
RHO_INF_SWEEP = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)

# The record is STOP_TIME long, so spectra come in 1/STOP_TIME = 50 Hz bins. That
# puts twelve bins between the two edges below, comfortably wider than the Kaiser
# main lobe, so the bulk response cannot leak into the measured high band.
LF_MAX = 400.0             # Hz; the physical response has died out by here
HF_MIN = 1000.0            # Hz; omega*dt > 0.6 here, well into the damped range

# Observed on this fixture, whose runs reproduce bit-for-bit: at RhoInf=1 the high
# band holds 1.4e-4 of the bulk, each 0.2 drop in RhoInf costs it at least 1.47x,
# and the ends of the sweep differ by 140x. The margins below are wide enough to
# absorb platform drift while still failing on a scheme that has stopped damping.
MIN_HF_FRACTION = 5e-5     # high-band energy over low-band energy, at RhoInf=1
MIN_HF_DECAY = 1.2         # per sweep step
MIN_HF_RANGE = 20.0        # RhoInf=1 against RhoInf=0

# Pointwise deviation of the volume trace from the RhoInf=1 run, as a fraction of
# that run's peak-to-peak swing. Observed worst case is 1.4%, at RhoInf=0.
MAX_BULK_DEVIATION = 0.03

pytestmark = [pytest.mark.mpi, pytest.mark.slow]


def _stage(wd, rho_inf):
    """Copy mesh and settings into wd, substituting the swept RhoInf.

    The fixture uses paths relative to the working dir, so the mesh and the
    ./Results output folder have to live next to the settings file.
    """
    (wd / "tetgen").mkdir()
    for f in SRC.glob("tetgen/ellipsoid.*"):
        shutil.copy(f, wd / "tetgen")
    text = FIXTURE.read_text()
    assert text.count(DEFAULT_RHO_INF) == 1, f"{FIXTURE.name}: no RhoInf element to substitute"
    settings = wd / FIXTURE.name
    settings.write_text(text.replace(DEFAULT_RHO_INF, f"<RhoInf>{rho_inf}</RhoInf>"))
    (wd / "Results").mkdir()
    return settings


def _volume_trace(wd):
    """Cavity volume per time step, asserting the sampling the spectra assume."""
    names, data = read_table(wd / "Results" / "Pressure.dat")
    t, v = data[:, names.index("time")], data[:, names.index("volume1")]
    # CardioMechanics exits 0 even when the solve gives up, and with MinTimeStep
    # pinned it gives up rather than sub-stepping, so a short trace is how a
    # diverged run shows itself.
    assert len(t) == EXPECTED_SAMPLES, (
        f"{wd}: trace has {len(t)} samples, expected {EXPECTED_SAMPLES}; the run "
        f"stopped at t={t[-1]:.4f} s instead of {STOP_TIME} s")
    # Also catches a repeated sample, which the exporter is capable of emitting.
    assert np.allclose(np.diff(t), DT), f"{wd}: trace is not sampled uniformly at dt={DT}"
    return v


def _band_energy(volume, low, high):
    """Energy of the volume trace between two frequencies, in Hz.

    The Kaiser window is what makes the high band readable at all: the bulk
    response is four orders of magnitude larger than the ringing, and the
    sidelobes of a plainer window bury the latter under leakage from the former.
    """
    n = len(volume)
    amplitude = np.abs(np.fft.rfft((volume - volume.mean()) * np.kaiser(n, 14)))
    f = np.fft.rfftfreq(n, DT)
    return float(np.linalg.norm(amplitude[(f >= low) & (f < high)]))


def _hf_energy(volume):
    return _band_energy(volume, HF_MIN, np.inf)


@pytest.fixture(scope="module")
def sweep(binary, cm_env, tmp_path_factory):
    """Run the ringdown fixture once per RhoInf. Returns {rho_inf: volume trace}."""
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not found")
    traces = {}
    for rho_inf in RHO_INF_SWEEP:
        wd = tmp_path_factory.mktemp(f"ringdown_{rho_inf}")
        settings = _stage(wd, rho_inf)
        run_binary(binary("CardioMechanics"), ["-settings", settings.name],
                   cwd=wd, env=cm_env, np=NP, timeout=1800)
        traces[rho_inf] = _volume_trace(wd)
    return traces


def test_step_load_excites_high_frequency_content(sweep):
    """Without excitation there is nothing for the sweep to damp.

    Measured inside the RhoInf=1 run alone, as a fraction of its own bulk
    response, so that this says something about the fixture rather than
    restating the endpoints of the sweep below.
    """
    volume = sweep[1.0]
    fraction = _hf_energy(volume) / _band_energy(volume, 0.0, LF_MAX)
    assert fraction >= MIN_HF_FRACTION, (
        f"at RhoInf=1 the band above {HF_MIN} Hz holds only {fraction:.2e} of the "
        f"energy below {LF_MAX} Hz, under the {MIN_HF_FRACTION:.0e} this fixture is "
        f"meant to excite: the step is no longer ringing the unresolved modes")


def test_high_frequency_content_decreases_with_rho_inf(sweep):
    """The point of the whole exercise: lower RhoInf, less high-frequency content."""
    hf = [(r, _hf_energy(sweep[r])) for r in RHO_INF_SWEEP]
    (hi_rho, hi_e), (lo_rho, lo_e) = min(zip(hf, hf[1:]),
                                         key=lambda pair: pair[0][1] / pair[1][1])
    sweep_report = ", ".join(f"{r}={e:.3e}" for r, e in hf)
    assert hi_e / lo_e >= MIN_HF_DECAY, (
        f"high-band energy above {HF_MIN} Hz did not fall by at least "
        f"{MIN_HF_DECAY}x between RhoInf={hi_rho} and RhoInf={lo_rho}: "
        f"{hi_e:.3e} vs {lo_e:.3e} (ratio {hi_e / lo_e:.3f}). Full sweep: {sweep_report}")
    ends = hf[0][1] / hf[-1][1]
    assert ends >= MIN_HF_RANGE, (
        f"the sweep spans only {ends:.1f}x in high-band energy, under the "
        f"{MIN_HF_RANGE}x expected across the full RhoInf range: {sweep_report}")


def test_bulk_response_is_preserved(sweep):
    """Selective damping, not global: the response itself survives the sweep.

    Compared against RhoInf=1, the member of the family that adds no numerical
    damping at all. The comparison is pointwise on the raw trace, which the high
    band influences only at the 1e-4 level, so what it bounds is the bulk motion.
    """
    reference = sweep[1.0]
    swing = np.ptp(reference)
    for rho_inf in RHO_INF_SWEEP[1:]:
        deviation = np.abs(sweep[rho_inf] - reference) / swing
        i = int(np.argmax(deviation))
        assert deviation[i] <= MAX_BULK_DEVIATION, (
            f"volume trace at RhoInf={rho_inf} deviates from the RhoInf=1 trace by "
            f"{deviation[i]:.4f} of the {swing:.4e} mL bulk swing, above "
            f"{MAX_BULK_DEVIATION}: sample {i} (t={i * DT:.4f} s) "
            f"actual={sweep[rho_inf][i]:.6e} reference={reference[i]:.6e}")

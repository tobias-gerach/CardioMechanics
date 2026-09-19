#!/usr/bin/env python3

"""Fit the passive material parameters of a CardioMechanics model to an end-diastolic
pressure-volume relation.

The module holds the optimizer's inputs - the settings file it edits, the material law that file
declares together with the parameters of that law which carry the stiffness level and the exponents,
and the cavity volume of the mesh the file names - and its arithmetic: the empirical Klotz
end-diastolic pressure-volume relation, the exponential model fitted to a pressure-volume curve, the
parameter scalings that follow from two such fits, the curve that fit is taken over, cut out of the
records the recovery plugin writes, and where the outer loop stands. Each of those is a function that
takes data and returns data, so the loop that drives the runs holds no decisions.

Volumes are in millilitres and pressures in mmHg throughout the arithmetic: the Klotz relation is
published in those units, and mmHg is the number a clinical source reports.

The settings file is read and written structurally rather than by substitution on a template, so a
file whose formatting the tool has never seen survives a round trip. CardioMechanics settings are
rootless - several elements sit side by side at the top of the document - so the text is wrapped in
a synthetic root before parsing and the root is dropped again on writing. Comments, processing
instructions and whitespace are kept, which is what makes the written file still readable by the
person who wrote it.

Parameters are addressed by the dotted key the solver itself uses, for instance
`Materials.Mat_30.Usyk.a`.

Run as a script, the module is the outer loop those functions serve: each iteration runs
CardioMechanics with the `ReferenceRecovery` plugin from the target geometry, fits the exponential
model to the pressure-volume curve of the last inflation and to the Klotz relation, and scales the
material parameters by the ratio of the two fits. Only the parameters carry forward. The loop ends on
convergence or on parameter stagnation, and fails - with a non-zero exit status and the path of the
run's output or log - on a run that did not finish, on a recovery that did not reach the target
configuration, and on the iteration cap.

Every run leaves in its work directory a record of each outer iteration, a figure overlaying the
Klotz relation with the simulated curves, and the recovered unloaded node file, which is the
deliverable: it is what the `LoadUnloadedState` plugin consumes.
"""

import argparse
import csv
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

# The stiffness parameters and the exponents of each law, by the names they carry in a settings
# file. Every stiffness parameter takes the same scaling and every exponent the same scaling, so a
# law's anisotropy ratios are properties of the model and not of the fit. The bulk modulus is a
# numerical incompressibility penalty rather than a property of the tissue and is in neither group;
# so are the dispersion of Holzapfel and the aScale and bScale factors.
LAW_PARAMETERS = {
    "Usyk": (("a",), ("bff", "bss", "bnn", "bfs", "bfn", "bns")),
    "Guccione": (("C",), ("bf", "bt", "bfs")),
    "Holzapfel": (("a", "af", "as", "afs"), ("b", "bf", "bs", "bfs")),
}

_ROOT = "CardioMechanicsSettings"
# An XML declaration may only stand at the head of a document, so it cannot go inside the synthetic
# root. It is carried as an attribute of that root, which no parameter key can reach.
_DECLARATION = re.compile(r"\s*<\?xml.*?\?>", re.S)
_PROLOGUE = "prologue"
_REQUIRED = object()


def read_settings(path):
    """The settings file at `path`, as the synthetic root element over its top-level elements."""
    text = Path(path).read_text()
    declaration = _DECLARATION.match(text)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    parser.feed(f"<{_ROOT}>{text[declaration.end():] if declaration else text}</{_ROOT}>")
    root = parser.close()
    root.set(_PROLOGUE, declaration.group() if declaration else "")
    return root


def write_settings(root, path):
    """Write the document under the synthetic root `root` to `path`, without that root."""
    # The solver's reader has no notion of a self-closing tag: it would read <X /> as a tag named
    # "X/" and then reject the closing tag it never opened.
    body = "".join(ET.tostring(child, encoding="unicode", short_empty_elements=False) for child in root)
    Path(path).write_text(root.get(_PROLOGUE, "") + (root.text or "") + body)


def _find(root, key):
    element = root
    for tag in key.split("."):
        element = element.find(tag)
        if element is None:
            return None
    return element


def get_parameter(root, key, default=_REQUIRED):
    """The text of the parameter `key`, or `default`. Raises KeyError when the file does not carry
    the parameter and no default is given. The text is returned as it stands, since the solver's own
    reader keeps everything between a value's first non-blank character and the next tag."""
    element = _find(root, key)
    if element is None:
        if default is _REQUIRED:
            raise KeyError(f"the settings file carries no {key}")
        return default
    return element.text or ""


def set_parameter(root, key, value):
    """Set the parameter `key` to `str(value)`. Raises KeyError when the file does not carry the
    parameter: a parameter the file leaves to the solver's default is one the tool cannot scale, and
    silently creating it would fit against a value the user never saw."""
    element = _find(root, key)
    if element is None:
        raise KeyError(f"the settings file carries no {key}")
    element.text = str(value)


def set_output(root, key, value):
    """Point the output `key` at `value`, creating the elements the file does not carry. An output
    name is the tool's to set, unlike a material parameter: the run belongs to the tool, and a name
    the file leaves to the solver's default would drop that run's output wherever it was started."""
    element = root
    for tag in key.split("."):
        child = element.find(tag)
        element = ET.SubElement(element, tag) if child is None else child
    element.text = str(value)


def material_law(root):
    """The constitutive law the settings file declares, and the key prefix of every material block
    that carries it, in document order. Raises ValueError when the blocks disagree on the law: the
    fit scales one law's parameters, and which of several it should scale is not the tool's
    choice."""
    materials = root.find("Materials")
    laws = {material.tag: material.find("Type").text
            for material in ([] if materials is None else materials)
            if isinstance(material.tag, str) and material.find("Type") is not None}
    if not laws:
        raise ValueError("the settings file declares no material with a Type")
    if len(set(laws.values())) > 1:
        raise ValueError(f"the materials declare more than one law: {laws}")
    law = next(iter(laws.values()))
    return law, [f"Materials.{material}.{law}" for material in laws]


def law_parameters(law, stiffness=None, exponents=None):
    """The names of the stiffness parameters and of the exponents of `law`, each overridable for a
    law LAW_PARAMETERS does not list. Raises ValueError for an unlisted law with no override."""
    if law not in LAW_PARAMETERS and (stiffness is None or exponents is None):
        raise ValueError(f"{law} is not in the parameter table {sorted(LAW_PARAMETERS)}; "
                         "name its stiffness and exponent parameters instead")
    listed = LAW_PARAMETERS.get(law, ((), ()))
    return (tuple(listed[0] if stiffness is None else stiffness),
            tuple(listed[1] if exponents is None else exponents))


def mesh_files(root, path):
    """The tetgen node and surface files the settings file at `path` names, and the length of one of
    their coordinate units in metres. A relative name is resolved against the directory of the
    settings file, which is where those names resolve when CardioMechanics runs."""
    directory = Path(path).parent
    return (directory / get_parameter(root, "Mesh.Tetgen.Nodes"),
            directory / get_parameter(root, "Mesh.Tetgen.Surfaces"),
            float(get_parameter(root, "Mesh.Tetgen.Unit", 1.0)))


def cavity_volume(nodes, surfaces, surface, unit):
    """The volume in millilitres enclosed by the faces of surface index `surface` of the tetgen mesh
    in the files `nodes` and `surfaces`, whose coordinates are in units of `unit` metres.

    The faces of a cavity point out of the cavity, so the volume is positive. Asserts that it does
    not depend on the point the tetrahedra spanning it are taken from, which holds exactly when the
    faces close - the criterion of CBCirculationCavity::ClosedSurfaceCheck - since a volume read off
    an open surface is a number about the origin rather than about the geometry."""
    coordinates = unit * np.loadtxt(nodes, skiprows=1, usecols=(1, 2, 3))
    rows = np.loadtxt(surfaces, skiprows=1, dtype=int, ndmin=2)
    # A face is its index, its nodes, then the attributes the header counts: the material index, the
    # surface index and optionally a traction scaling. The solver reads the node count off the row
    # rather than off the header, so the row and the attribute count fix where the surface index is.
    with open(surfaces) as file:
        attributes = int(file.readline().split()[2])
    assert attributes >= 2, f"{surfaces} carries no surface index, only {attributes} attribute(s)"
    per_face = rows.shape[1] - attributes - 1
    faces = rows[rows[:, per_face + 2] == surface][:, 1:per_face + 1] - 1
    assert len(faces), f"no face of the mesh carries surface index {surface}"
    # The solver spans a face by a fan from its first node, over the polygon through all of its
    # nodes: the triangle itself for a three-node face, the hexagon of vertices and mid-edge nodes
    # for a six-node one.
    polygon = coordinates[faces][:, [0, 3, 1, 4, 2, 5] if faces.shape[1] == 6 else [0, 1, 2]]
    a, b, c = polygon[:, :1], polygon[:, 1:-1], polygon[:, 2:]
    areas = np.cross(b - a, c - a)

    def volume(reference):
        return 1e6 * np.einsum("fkj,fkj->", a - reference, areas) / 6

    enclosed = volume(np.zeros(3))
    assert abs(volume(np.ones(3)) - enclosed) <= 1e-10, f"surface {surface} does not close"
    return enclosed


# The plugin writes its pressures in pascal and its volumes in millilitres, while the arithmetic here
# works in mmHg, so the records are converted on the way in.
PASCAL_PER_MMHG = 133.322387415

Inflation = namedtuple("Inflation", "volumes pressures residual_norm cycles")


def _columns(record):
    """The columns of a whitespace-separated record with one header line, by their header names."""
    lines = [line for line in record.splitlines() if line.strip()]
    names = lines[0].split() if lines else []
    rows = np.array([[float(field) for field in line.split()] for line in lines[1:]])
    assert len(rows), f"the record carries no data, only the header {' '.join(names)!r}"
    assert rows.shape[1] == len(names), f"{rows.shape[1]} columns under {len(names)} header names"
    return dict(zip(names, rows.T))


def last_inflation(pressure_volume_record, cycle_record, surface=1):
    """The pressure-volume curve of the last complete inflation of cavity `surface`, in volumes in ml
    and pressures in mmHg, which is the curve the fit is taken over, together with the infinity norm
    in metres of the nodal residual the recovery ended at - the norm the plugin's own tolerance is on,
    so the caller can tell whether the recovery reached the target configuration - and the number of
    inner cycles it took, which is one more than the zero-based counter the record carries.

    Each inner cycle of the recovery inflates its current guess of the unloaded configuration once,
    and the cycle record carries the time each of those inflations finished. The curve is therefore
    cut out of the pressure-volume record by those times rather than by counting rows, so a run whose
    inner loop converged in a single cycle needs no separate treatment, and rows of an inflation whose
    cycle never finished - a run that was cut off - fall outside the last boundary and are dropped.

    The first row of an inflation stands at the first pressure increment rather than at zero, since
    the plugin writes a row per time step of the ramp. It is replaced by the unloaded volume the cycle
    record carries at zero pressure, which is the configuration that inflation started from, so the
    curve is anchored where the exponential model carries no pressure. The rows are otherwise returned
    as they stand, including the target pressure that the first cycle alone carries twice, since a
    repeated measurement is not a wrong one."""
    cycles = _columns(cycle_record)
    record = _columns(pressure_volume_record)
    boundaries, time = cycles["Time"], record["Time"]
    window = time <= boundaries[-1]
    if len(boundaries) > 1:
        window &= time > boundaries[-2]
    volumes = record[f"Volume{surface}"][window]
    pressures = record[f"Pressure{surface}"][window] / PASCAL_PER_MMHG
    assert len(volumes) > 1, f"the last inflation of surface {surface} holds fewer than two rows"
    volumes[0], pressures[0] = cycles[f"UnloadedVolume{surface}"][-1], 0.0
    return Inflation(volumes, pressures, cycles["ResidualNorm"][-1], int(cycles["Cycle"][-1]) + 1)


# Klotz et al. (2006), the empirical end-diastolic pressure-volume relation, in mmHg and ml. It is
# built in two steps. First the measured pair fixes two volumes: the unloaded volume, a published
# affine fraction of the measured volume, and V30, the volume at KLOTZ_P30, reached through the
# population regression p = KLOTZ_AN * ((V - V0) / (V30 - V0)) ** KLOTZ_BN. The relation itself is
# then the power law p = alpha * V ** beta through the two anchors (V30, KLOTZ_P30) and the measured
# pair, so it carries the measurement exactly and the regression only as far as V30.
KLOTZ_AN, KLOTZ_BN = 27.78, 2.76
KLOTZ_V0_INTERCEPT, KLOTZ_V0_SLOPE = 0.6, 0.006
KLOTZ_P30 = 30.0

# Both parameter scalings are clamped to this interval. A fit that asks for more than a fivefold
# move in one outer iteration is extrapolating far outside the pressure range it saw, and a forward
# solve with such parameters diverges rather than informing the next iteration.
SCALING_BOUNDS = (0.2, 5.0)
# Convergence of the outer loop: both volume residuals are reported against the same half a percent of
# the measured end-diastolic volume, and the loop has stagnated once no scaling asks for a move of
# more than a tenth of a percent.
VOLUME_TOLERANCE = 0.005
STAGNATION_TOLERANCE = 1e-3

Scalings = namedtuple("Scalings", "stiffness exponent clamped")
Convergence = namedtuple("Convergence", "unloaded end_diastolic stagnation converged stagnated")


def klotz_volumes(pressure, volume):
    """The unloaded volume and the volume at KLOTZ_P30, in ml, that the Klotz relation predicts from
    the measured pair `pressure` in mmHg and `volume` in ml."""
    assert pressure > 0 and volume > 0, f"{pressure} mmHg in {volume} ml is not an end-diastolic pair"
    unloaded = volume * (KLOTZ_V0_INTERCEPT - KLOTZ_V0_SLOPE * pressure)
    return unloaded, unloaded + (volume - unloaded) / (pressure / KLOTZ_AN) ** (1 / KLOTZ_BN)


def klotz_coefficients(pressure, volume):
    """The coefficients alpha and beta of the Klotz relation p = alpha * V ** beta predicted from the
    measured pair, with p in mmHg and V in ml. They are fixed by the two anchors the relation is
    built on, so the curve carries the measured pair and KLOTZ_P30 at V30 exactly."""
    v30 = klotz_volumes(pressure, volume)[1]
    beta = np.log(pressure / KLOTZ_P30) / np.log(volume / v30)
    return KLOTZ_P30 / v30 ** beta, beta


def klotz_pressure(volumes, alpha, beta):
    """The Klotz pressure in mmHg at `volumes` in ml, given the coefficients of the relation."""
    return alpha * np.asarray(volumes, float) ** beta


def klotz_curve(pressure, volume, samples=100):
    """The Klotz relation predicted from the measured pair, as volumes in ml and pressures in mmHg,
    sampled from the unloaded volume to V30. That is the range the relation is built over, and so the
    range the exponential model is fitted across."""
    unloaded, v30 = klotz_volumes(pressure, volume)
    volumes = np.linspace(unloaded, v30, samples)
    return volumes, klotz_pressure(volumes, *klotz_coefficients(pressure, volume))


def exponential_pressure(volumes, unloaded, prefactor, exponent):
    """The exponential pressure-volume model, in mmHg at `volumes` in ml. The volume enters as its
    dilation from the unloaded volume, where the model carries zero pressure, so the prefactor is a
    pressure and the exponent is dimensionless. That is what lets the scalings compare the fits of
    two curves that do not share a size."""
    volumes = np.asarray(volumes, float)
    return prefactor * (np.exp(exponent * (volumes - unloaded) / unloaded) - 1.0)


def fit_exponential(volumes, pressures, unloaded):
    """The prefactor in mmHg and the dimensionless exponent of the exponential model fitted to the
    pressure-volume curve (`volumes` in ml, `pressures` in mmHg) anchored at `unloaded` in ml."""
    volumes, pressures = np.asarray(volumes, float), np.asarray(pressures, float)
    dilation = volumes.max() / unloaded - 1.0
    assert dilation > 0, "the curve reaches no volume above the unloaded one"
    # An exponent spending one e-fold over the curve's dilation, and the prefactor that then carries
    # the curve's peak pressure. Far enough from a flat start that the fit sees curvature to work on.
    guess = (pressures.max() / np.expm1(1.0), 1.0 / dilation)
    (prefactor, exponent), _ = curve_fit(
        lambda v, a, b: exponential_pressure(v, unloaded, a, b), volumes, pressures, p0=guess)
    return prefactor, exponent


def parameter_scalings(klotz_fit, simulated_fit):
    """The factors the stiffness parameter and every exponent parameter are multiplied by, from the
    exponential fits to the Klotz relation and to the simulated curve, each as (prefactor, exponent).

    One factor for all exponents preserves the law's anisotropy ratios: how much stiffer the tissue
    is along the fibres than across them is a property of the model and not of this fit. The
    magnitude of the simulated exponent is used, so that a simulated curve flat enough for the fit to
    return a negative exponent still yields a positive factor and stiffens rather than inverts the
    law. Both factors are clamped to SCALING_BOUNDS, and the names of those that were clamped are
    returned so the caller can say when the fit asked for a jump the algorithm refuses to take."""
    klotz_prefactor, klotz_exponent = klotz_fit
    simulated_prefactor, simulated_exponent = simulated_fit
    # A negative prefactor is a fit that found no rising pressure-volume curve at all, not a fit
    # asking for a large move; clamping it would soften the law by the full bound and report that as
    # an over-eager fit.
    assert simulated_prefactor > 0, f"the simulated fit has no positive prefactor: {simulated_fit}"
    asked = (klotz_prefactor / simulated_prefactor, klotz_exponent / abs(simulated_exponent))
    clamped = tuple(name for name, factor in zip(("stiffness", "exponent"), asked)
                    if not SCALING_BOUNDS[0] <= factor <= SCALING_BOUNDS[1])
    return Scalings(*np.clip(asked, *SCALING_BOUNDS), clamped)


def convergence(unloaded, klotz_unloaded, end_diastolic, measured, scalings):
    """Where the outer loop stands, from the simulated unloaded and end-diastolic volumes against the
    Klotz unloaded volume and the measured volume, all in ml, and the scalings the fit just asked for.

    The unloaded-volume residual is what the fit is driving down, and it alone decides convergence.
    The end-diastolic-volume residual is near-tautological - the recovery drives the loaded
    configuration onto the target to within the plugin's own tolerance - and is a sanity check on the
    inner loop rather than a criterion the fit can influence: it carries whatever bias the inner loop
    leaves, so holding the fit to it would report a well-fitted material as unconverged. Stagnation is
    reported separately from convergence, so a run that stopped moving is not mistaken for one that
    met its target. Both volume residuals are in ml and are held against one tolerance, a fraction of
    the measured end-diastolic volume, which is the one length scale of the problem that does not move
    between iterations."""
    unloaded_residual = abs(unloaded - klotz_unloaded)
    stagnation = max(abs(scalings.stiffness - 1.0), abs(scalings.exponent - 1.0))
    return Convergence(unloaded_residual, abs(end_diastolic - measured), stagnation,
                       converged=unloaded_residual <= VOLUME_TOLERANCE * measured,
                       stagnated=stagnation <= STAGNATION_TOLERANCE)


# What one outer iteration leaves in its own directory. Every run is given a directory of its own, so
# the records read back after a run are unambiguously that run's, and nothing a long loop produces
# lands in the tree the settings file was read from.
LOG_NAME = "CardioMechanics.log"
OUTPUT_NAME = "CardioMechanics.out"
RECOVERY_DIR = "ReferenceRecovery"
EXPORT_DIR = "Export"
# Enough digits that a scaling of the size the loop applies is visible in the written settings file,
# few enough that the file stays readable after a dozen iterations.
PARAMETER_FORMAT = ".6g"


def retarget_outputs(root, directory):
    """`root` with its outputs retargeted at `directory`: the log, the records the recovery plugin
    writes and, if the file asks for one, the exported time series.

    The inputs the file names are left as they stand, and the run is given the settings file's own
    directory to work in, so that a relative mesh name resolves the way it does for a hand-started run
    and the tool needs no list of which settings keys name a file.

    The log and the plugin's export directory are set whether or not the file carries them: the loop
    names that log when a run fails and reads the pressure-volume record out of that directory. A
    prefix for the exported time series is only moved, never created, so that the tool does not turn
    on an export the user's file left off. Only the basename of that prefix is kept, so retargeting a
    document a second time does not stack one iteration's directory on the next."""
    directory = Path(directory)
    set_output(root, "General.LogFile", directory / LOG_NAME)
    set_output(root, "Plugins.ReferenceRecovery.ExportDir", directory / RECOVERY_DIR)
    prefix = Path(get_parameter(root, "Export.Prefix", "").strip()).name
    if prefix:
        set_parameter(root, "Export.Prefix", directory / EXPORT_DIR / prefix)
    return root


def parameter_keys(blocks, stiffness, exponents):
    """The keys of the parameters the optimizer scales, in a fixed order, so that a record of several
    iterations carries one column per parameter."""
    return [f"{block}.{name}" for block in blocks for name in (*stiffness, *exponents)]


def scaled_parameters(root, blocks, stiffness, exponents, scalings):
    """The material parameters of every block of `blocks` after one scaling step, as their keys and
    their new values. The stiffness parameters named in `stiffness` take the stiffness scaling and the
    exponents named in `exponents` the exponent scaling; every other parameter of the law, the bulk
    modulus among them, is left alone."""
    return {f"{block}.{name}": float(get_parameter(root, f"{block}.{name}")) * factor
            for block in blocks
            for names, factor in ((stiffness, scalings.stiffness), (exponents, scalings.exponent))
            for name in names}


def reached_target(inflation, tolerance):
    """Whether the recovery arrived at the target configuration, from the inflation it ended on and
    the plugin's own tolerance in metres. The plugin writes its unloaded node file when it stops on
    its cycle cap just as it does when it converges, so the existence of that file says nothing; the
    residual the tolerance is on is what says it."""
    return inflation.residual_norm <= tolerance


# The three outputs sit in the work directory itself, beside the per-iteration directories, and all
# three are rewritten after every iteration, so a loop that a later iteration ends still leaves what
# the ones before it produced.
ITERATIONS_CSV = "iterations.csv"
FIGURE_NAME = "PressureVolume.png"
UNLOADED_NAME = "UnloadedState.node"


def iteration_row(iteration, parameters, inflation, state, scalings, walltime, corrupt):
    """One row of the iteration record: the volumes and residuals the run produced, the wall time it
    took, the corrupt-element reports it printed and the material parameters it was given. What the
    fit asks for next is named for the iteration it applies to rather than the one that produced it,
    so that a reader of the record is not left to infer which side of a scaling a row stands on."""
    return {"iteration": iteration, "cycles": inflation.cycles,
            "residual_m": inflation.residual_norm, "unloaded_ml": inflation.volumes[0],
            "end_diastolic_ml": inflation.volumes[-1], "unloaded_residual_ml": state.unloaded,
            "end_diastolic_residual_ml": state.end_diastolic,
            "next_stiffness_scaling": scalings.stiffness,
            "next_exponent_scaling": scalings.exponent,
            "next_parameter_move": state.stagnation, "next_clamped": " ".join(scalings.clamped),
            "walltime_s": walltime, "corrupt_reports": corrupt, **parameters}


def write_iterations(path, rows):
    """The iteration record at `path`. The columns follow the first row, so the parameters carry the
    names the settings file gives them and a law with other parameters needs nothing here."""
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(path, klotz, klotz_fit, inflations):
    """The figure at `path`: the Klotz relation, the exponential model fitted to it and the simulated
    pressure-volume curve of every outer iteration, over volume in ml and pressure in mmHg. The
    iterations run dark to light in the order they were run, so the figure reads as the material
    parameters moving the simulated relation onto the empirical one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    volumes, pressures = klotz
    figure, axes = plt.subplots(figsize=(5.4, 4.0), layout="constrained")
    axes.plot(volumes, pressures, color="0.1", lw=1.6, label="Klotz EDPVR")
    axes.plot(volumes, exponential_pressure(volumes, volumes[0], *klotz_fit), color="0.1", lw=1.0,
              ls="--", label="exponential fit")
    shades = matplotlib.colormaps["viridis"](np.linspace(0.15, 0.85, len(inflations)))
    for iteration, (inflation, shade) in enumerate(zip(inflations, shades), 1):
        axes.plot(inflation.volumes, inflation.pressures, color=shade, lw=1.2, marker="o", ms=3,
                  label=f"iteration {iteration}")
    # The Klotz relation is sampled to 30 mmHg, which is the range it is fitted over, while a run
    # stops at the target pressure. The axes are held to what the runs reached, since that is the
    # range over which the two curves are compared.
    right = max(max(inflation.volumes) for inflation in inflations)
    top = max(max(inflation.pressures) for inflation in inflations)
    axes.set(xlabel="cavity volume (ml)", ylabel="pressure (mmHg)",
             xlim=(None, 1.02 * right), ylim=(-0.03 * top, 1.10 * top))
    axes.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def recovered_node_file(records):
    """The unloaded node file a recovery run left in `records`. The plugin writes one per pressure
    increment, so the highest increment carries the recovered reference configuration."""
    files = sorted(Path(records).glob("UnloadedState_Incr*.node"),
                   key=lambda path: int(path.stem.rpartition("Incr")[2]))
    assert files, f"the recovery wrote no unloaded node file in {records}"
    return files[-1]


# The solver's runtime estimate, and the report a solid element makes from whichever rank holds it
# when the material law finds det F <= 0 at a quadrature point. Neither is anchored at the start of
# the line, so a launcher that prefixes each line with its rank does not hide them.
PROGRESS_LINE = re.compile(r"([-+.\deE]+)% done !!.*Estimated remaining time: (\d+:\d\d:\d\d)")
CORRUPT_REPORT = re.compile(r"Element with index \d+ is corrupt\.")


def follow_output(lines, record, show):
    """Write every line of a run's output `lines` to `record`, pass the solver's progress to `show`
    as a short text, and return the number of corrupt-element reports among the lines.

    Each report is the line search meeting an inverted element and cutting the step back. The run
    recovers from them, so they end nothing, but a count of them tells a fit that asked for more
    than the geometry takes apart from one that did not."""
    corrupt = 0
    for line in lines:
        record.write(line)
        progress = PROGRESS_LINE.search(line)
        if progress:
            show(f"{float(progress[1]):.0f}% done, {progress[2]} remaining")
        corrupt += len(CORRUPT_REPORT.findall(line))
    return corrupt


def run_cardiomechanics(binary, settings, directory, ranks, output):
    """Run `binary` on the settings file `settings` from the working directory `directory`, on
    `ranks` ranks, and return the number of corrupt-element reports it printed. Raises SystemExit
    naming `output` when the run fails, so that a diverged forward solve is debugged from what it
    printed rather than read as a fit.

    The run's standard output goes to `output` rather than to the terminal, where the thousands of
    lines of an iteration would scroll the loop's own lines away within seconds. The file is a
    superset of the solver's log: it also carries the per-rank corrupt-element reports, which reach
    no log. The terminal keeps a single line of the solver's progress, redrawn in place, so a run in
    flight can be told from a hung one; it is cleared when the run ends. The standard error is left
    on the terminal, since what reaches it is an error of the solver or of the launcher."""
    command = (["mpirun", "-np", str(ranks)] if ranks > 1 else []) + \
        [str(binary), "-settings", str(settings)]
    label = Path(output).parent.name

    def show(progress):
        print(f"\r{label}: {progress}\x1b[K", end="", flush=True)

    with open(output, "w") as record, subprocess.Popen(
            command, cwd=str(directory), stdout=subprocess.PIPE, text=True) as run:
        corrupt = follow_output(run.stdout, record, show)
    print("\r\x1b[K", end="", flush=True)
    if run.returncode:
        raise SystemExit(f"CardioMechanics exited {run.returncode}; its output is {output}")
    return corrupt


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        prog="FitMaterialParameters",
        description="Fit the passive material parameters of a CardioMechanics model so that its "
                    "end-diastolic pressure-volume relation matches the empirical Klotz relation "
                    "predicted from one measured pressure-volume pair. Each outer iteration runs "
                    "CardioMechanics with the ReferenceRecovery plugin from the target geometry; "
                    "only the material parameters carry forward.")
    parser.add_argument("settings", help="The CardioMechanics settings file to fit. It is never "
                                         "written: each run gets its own copy, carrying that "
                                         "iteration's parameters, under the work directory.")
    parser.add_argument("--pressure", type=float, required=True,
                        help="The measured end-diastolic pressure in mmHg, as a clinical source "
                             "reports it.")
    parser.add_argument("--volume", type=float,
                        help="The measured end-diastolic volume in ml. Defaults to the cavity volume "
                             "of the mesh the settings file names.")
    parser.add_argument("--surface", type=int, default=1,
                        help="The surface index of the cavity (default: %(default)s).")
    parser.add_argument("--iterations", type=int, default=10,
                        help="The most outer iterations to run (default: %(default)s).")
    parser.add_argument("--ranks", type=int, default=1,
                        help="MPI ranks per run (default: %(default)s).")
    parser.add_argument("--work-dir", default="FitMaterialParameters",
                        help="Where every run's settings file, log and output go "
                             "(default: %(default)s).")
    parser.add_argument("--binary", default="CardioMechanics",
                        help="The CardioMechanics binary (default: %(default)s).")
    parser.add_argument("--stiffness", nargs="+",
                        help="The names of the law's stiffness parameters, for a law the table does "
                             "not list.")
    parser.add_argument("--exponents", nargs="+",
                        help="The names of the law's exponent parameters, for a law the table does "
                             "not list.")
    arguments = parser.parse_args(argv)
    if arguments.iterations < 1:
        parser.error("--iterations must run at least one iteration")
    return arguments


def main(argv=None):
    """Run the outer loop. Ends quietly on convergence or on parameter stagnation, and raises
    SystemExit on a failed run, on a recovery that did not reach the target configuration, and on the
    iteration cap: each of those leaves parameters that were not fitted to what was asked for."""
    arguments = parse_arguments(argv)
    settings = Path(arguments.settings).resolve()
    root = read_settings(settings)
    law, blocks = material_law(root)
    stiffness, exponents = law_parameters(law, arguments.stiffness, arguments.exponents)
    nodes, surfaces, unit = mesh_files(root, settings)
    measured = arguments.volume
    if measured is None:
        measured = cavity_volume(nodes, surfaces, arguments.surface, unit)
    tolerance = float(get_parameter(root, "Plugins.ReferenceRecovery.Tolerance", 1e-3))

    klotz_unloaded = klotz_volumes(arguments.pressure, measured)[0]
    klotz = klotz_curve(arguments.pressure, measured)
    klotz_fit = fit_exponential(*klotz, klotz_unloaded)
    print(f"{law} against Klotz from {arguments.pressure:g} mmHg in {measured:.2f} ml: "
          f"unloaded volume {klotz_unloaded:.2f} ml, "
          f"tolerance {VOLUME_TOLERANCE * measured:.2f} ml")

    work = Path(arguments.work_dir).resolve()
    keys = parameter_keys(blocks, stiffness, exponents)
    rows, inflations = [], []
    try:
        for iteration in range(1, arguments.iterations + 1):
            directory = work / f"iteration_{iteration:02d}"
            (directory / EXPORT_DIR).mkdir(parents=True, exist_ok=True)
            current = directory / settings.name
            write_settings(retarget_outputs(root, directory), current)
            started = time.perf_counter()
            corrupt = run_cardiomechanics(arguments.binary, current, settings.parent,
                                          arguments.ranks, directory / OUTPUT_NAME)
            walltime = time.perf_counter() - started

            records = directory / RECOVERY_DIR
            inflation = last_inflation((records / "PressureVolumeInfo.dat").read_text(),
                                       (records / "CycleInfo.dat").read_text(), arguments.surface)
            if not reached_target(inflation, tolerance):
                raise SystemExit(
                    f"iteration {iteration}: the recovery stopped after {inflation.cycles} cycles at a "
                    f"residual of {inflation.residual_norm:.3e} m, above the plugin's tolerance of "
                    f"{tolerance:.3e} m. Its pressure-volume curve does not end at the target geometry "
                    f"and nothing is fitted to it; see {directory / LOG_NAME}")

            # A row of the pressure-volume record pairs the pressure a step was solved at with the
            # volume that step produced, so the curve is fitted as it stands. What the last row does
            # carry is the inner loop's own bias: the recovery drives its residual onto the
            # configuration one step behind, so the end-diastolic volume overshoots the target by
            # about one step of the ramp. That is the bias the end-diastolic residual reports and the
            # fit cannot influence.
            simulated_fit = fit_exponential(inflation.volumes, inflation.pressures,
                                            inflation.volumes[0])
            scalings = parameter_scalings(klotz_fit, simulated_fit)
            state = convergence(inflation.volumes[0], klotz_unloaded, inflation.volumes[-1], measured,
                                scalings)
            parameters = {key: float(get_parameter(root, key)) for key in keys}
            rows.append(iteration_row(iteration, parameters, inflation, state, scalings, walltime,
                                      corrupt))
            inflations.append(inflation)
            write_iterations(work / ITERATIONS_CSV, rows)
            plot_curves(work / FIGURE_NAME, klotz, klotz_fit, inflations)
            shutil.copyfile(recovered_node_file(records), work / UNLOADED_NAME)

            print(f"iteration {iteration}: {inflation.cycles} cycles at a residual of "
                  f"{inflation.residual_norm:.3e} m, unloaded {inflation.volumes[0]:.2f} ml "
                  f"(residual {state.unloaded:.2f} ml), end-diastolic {inflation.volumes[-1]:.2f} ml "
                  f"(residual {state.end_diastolic:.2f} ml), stiffness x{scalings.stiffness:.3f}, "
                  f"exponents x{scalings.exponent:.3f}, {corrupt} corrupt-element reports")
            if scalings.clamped:
                print(f"  the {' and '.join(scalings.clamped)} scaling the fit asked for was outside "
                      f"{SCALING_BOUNDS} and was clamped")
            if state.converged:
                print(f"converged: the unloaded volume is within "
                      f"{VOLUME_TOLERANCE * measured:.2f} ml of the Klotz prediction. "
                      f"The parameters are in {current}")
                break
            if state.stagnated:
                print(f"stagnated: no parameter moved by more than {STAGNATION_TOLERANCE:g}, with the "
                      f"unloaded volume still {state.unloaded:.2f} ml from the Klotz prediction. "
                      f"The parameters are in {current}")
                break
            for key, value in scaled_parameters(root, blocks, stiffness, exponents, scalings).items():
                set_parameter(root, key, format(value, PARAMETER_FORMAT))
        else:
            raise SystemExit(f"the loop reached its cap of {arguments.iterations} outer iterations "
                             f"without converging; the parameters of the last one are in {current}")

    finally:
        # Every ending names the outputs, the failures among them: an iteration that ran left its
        # record, its figure and its recovered configuration behind whatever stopped the loop.
        if rows:
            print(f"the recovered unloaded configuration is {work / UNLOADED_NAME}, the iterations "
                  f"are recorded in {work / ITERATIONS_CSV} and plotted in {work / FIGURE_NAME}")


if __name__ == "__main__":
    main()

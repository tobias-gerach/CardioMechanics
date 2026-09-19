"""Fit the passive material parameters of a CardioMechanics model to an end-diastolic
pressure-volume relation.

The module holds the optimizer's inputs - the settings file it edits, the material law that file
declares together with the parameters of that law which carry the stiffness level and the exponents,
and the cavity volume of the mesh the file names - and its arithmetic: the empirical Klotz
end-diastolic pressure-volume relation, the exponential model fitted to a pressure-volume curve, the
parameter scalings that follow from two such fits, and where the outer loop stands. Each of those is
a function that takes data and returns data, so the loop that drives the runs holds no decisions.

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
"""

import re
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
# Convergence of the outer loop: both volume residuals are held against the same half a percent of
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

    The unloaded-volume residual is what the fit is driving down. The end-diastolic-volume residual
    is near-tautological - the recovery drives the loaded configuration onto the target to within the
    plugin's own tolerance - and is reported as a sanity check on the inner loop rather than as a
    criterion the fit can influence. Stagnation is reported separately from convergence, so a run
    that stopped moving is not mistaken for one that met its target. Both volume residuals are in ml
    and are held against one tolerance, a fraction of the measured end-diastolic volume, which is the
    one length scale of the problem that does not move between iterations."""
    tolerance = VOLUME_TOLERANCE * measured
    residuals = (abs(unloaded - klotz_unloaded), abs(end_diastolic - measured))
    stagnation = max(abs(scalings.stiffness - 1.0), abs(scalings.exponent - 1.0))
    return Convergence(*residuals, stagnation,
                       converged=all(residual <= tolerance for residual in residuals),
                       stagnated=stagnation <= STAGNATION_TOLERANCE)

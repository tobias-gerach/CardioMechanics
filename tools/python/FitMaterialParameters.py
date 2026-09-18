"""Fit the passive material parameters of a CardioMechanics model to an end-diastolic
pressure-volume relation.

So far this module holds the optimizer's inputs alone: the settings file it edits, the material law
that file declares together with the parameters of that law which carry the stiffness level and the
exponents, and the cavity volume of the mesh the file names. The arithmetic that fits the parameters
and the loop that drives the runs come on top of it.

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
from pathlib import Path

import numpy as np

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

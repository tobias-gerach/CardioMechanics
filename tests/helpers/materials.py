"""Material laws shared by the verification tests against closed-form solutions."""

# Settings of each law, with the tag of its bulk modulus; every modulus is O(1). Holzapfel's k is
# the slope of its smoothed Heavyside switch. At 10 rather than the default 100, the fibre I4 of
# 1.25 along the load and 0.80 across it in the uniaxial patch test lies on the flank of the
# switch, where its derivative adds about 2e-2 to the fibre and sheet stress; at 100 it would add
# at most 7e-5.
LAWS = {
    "NeoHooke": ("k", {"a": 1}),
    "Holzapfel": ("kappa", {"a": 1, "b": 1, "af": 1, "bf": 1, "as": 0.5, "bs": 1, "afs": 0.3, "bfs": 1,
                            "k": 10}),
    "Guccione": ("K", {"C": 1, "bf": 8, "bt": 2, "bfs": 4}),
}


def material_block(law, kappa):
    """The XML settings block of law with bulk modulus kappa."""
    tag, params = LAWS[law]
    return f"<{law}>" + "".join(f"<{k}>{v}</{k}>" for k, v in {**params, tag: kappa}.items()) + f"</{law}>"

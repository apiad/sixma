from sixma import certify, generators as g

# NO typing.Annotated needed!


@certify(reliability=0.9, confidence=0.9)
def test_clean_syntax(
    x: g.Integer(0, 10),
    y = g.Integer(11, 20),
):
    # This should work perfectly
    assert x < y
    assert isinstance(x, int)

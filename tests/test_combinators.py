import math
from dataclasses import dataclass
from typing import Annotated
from sixma import certify, generators as g


# --- Helper Class for Object Testing ---
@dataclass
class User:
    id: int
    username: str
    is_active: bool


# --- Tests ---


@certify(reliability=0.9, confidence=0.95)
def test_list_combinator(
    # Generates a list of integers between 10 and 20.
    # List length must be between 2 and 5.
    items: Annotated[list[int], g.List(g.Integer(10, 20), min_len=2, max_len=5)],
):
    """Verifies that List generator respects length and element constraints."""
    assert isinstance(items, list)
    assert 2 <= len(items) <= 5

    for x in items:
        assert isinstance(x, int)
        assert 10 <= x <= 20


@certify(reliability=0.9, confidence=0.95)
def test_dict_combinator(
    # Generates a dictionary with specific keys and value types
    config: Annotated[
        dict,
        g.Dict(
            host=g.String(max_len=10), port=g.Integer(8000, 9000), ssl=g.Bool()
        ),
    ],
):
    """Verifies that Dict generator produces the correct schema."""
    assert isinstance(config, dict)
    assert set(config.keys()) == {"host", "port", "ssl"}

    assert isinstance(config["host"], str)
    assert len(config["host"]) <= 10

    assert isinstance(config["port"], int)
    assert 8000 <= config["port"] <= 9000

    assert isinstance(config["ssl"], bool)


@certify(reliability=0.9, confidence=0.95)
def test_object_combinator(
    # Generates instances of the User dataclass
    user: Annotated[
        User,
        g.Object(
            User,
            id=g.Integer(1, 1000),
            username=g.String(max_len=8),
            is_active=g.Bool(),
        ),
    ],
):
    """Verifies that Object generator instantiates classes correctly."""
    assert isinstance(user, User)

    assert 1 <= user.id <= 1000
    assert len(user.username) <= 8
    assert isinstance(user.is_active, bool)


@certify(reliability=0.9, confidence=0.95)
def test_nested_combinators(
    # Complex nested structure: A list of dicts
    # List[Dict[str, int]]
    matrix: Annotated[
        list[dict],
        g.List(
            g.Dict(x=g.Integer(0, 10), y=g.Integer(0, 10)), min_len=1, max_len=3
        ),
    ],
):
    """Verifies that combinators can be nested arbitrarily."""
    assert isinstance(matrix, list)
    assert 1 <= len(matrix) <= 3

    for point in matrix:
        assert isinstance(point, dict)
        assert 0 <= point["x"] <= 10
        assert 0 <= point["y"] <= 10


@certify(reliability=0.9, confidence=0.95)
def test_float_special_values(
    # Test generation of NaN and Infinity
    val: Annotated[float, g.Float(0.0, 1.0, allow_nan=True, allow_inf=True)],
):
    """Verifies Float generator handles special IEEE 754 values."""
    if math.isnan(val):
        return  # NaN is valid
    if math.isinf(val):
        return  # Inf is valid

    assert 0.0 <= val <= 1.0

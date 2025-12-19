from datetime import date, datetime
from dataclasses import dataclass
from typing import List, Dict
from types import SimpleNamespace

from sixma import certify, generators as g

# --- Helper Objects ---


@dataclass
class User:
    id: int
    username: str


# --- Tests ---


@certify(reliability=0.9, confidence=0.9)
def test_primitives_syntax(
    # Clean Syntax: Type hint says 'int', default value provides the generator
    i: int = g.Integer(0, 100),
    f: float = g.Float(0.0, 1.0),
    b: bool = g.Bool(),
    s: str = g.String(max_len=10),
):
    """
    Verifies that primitives work with the default-value syntax.
    """
    assert isinstance(i, int)
    assert 0 <= i <= 100

    assert isinstance(f, float)
    assert 0.0 <= f <= 1.0

    assert isinstance(b, bool)
    assert isinstance(s, str)
    assert len(s) <= 10


@certify(reliability=0.9, confidence=0.9)
def test_structures_syntax(
    # Type hint says List[int], default value is the List generator
    numbers: List[int] = g.List(g.Integer(0, 10), min_len=1, max_len=5),
    # Type hint says Dict, default is Dict generator
    profile: Dict[str, str] = g.Dict(
        name=g.String(max_len=5), role=g.String(max_len=5)
    ),
):
    """
    Verifies that List and Dict combinators work with default values.
    """
    assert isinstance(numbers, list)
    assert 1 <= len(numbers) <= 5
    for n in numbers:
        assert isinstance(n, int)
        assert 0 <= n <= 10

    assert isinstance(profile, dict)
    assert "name" in profile
    assert "role" in profile
    assert isinstance(profile["name"], str)


@certify(reliability=0.9, confidence=0.9)
def test_object_syntax(
    # Type hint says User (dataclass), default is Object generator
    user: User = g.Object(User, id=g.Integer(1000, 9999), username=g.String(max_len=8))
):
    """
    Verifies that Object generator injects the correct class instance.
    """
    assert isinstance(user, User)
    assert 1000 <= user.id <= 9999
    assert len(user.username) <= 8


@certify(reliability=0.9, confidence=0.9)
def test_temporal_syntax(
    # Type hint says date/datetime
    d: date = g.Date(date(2023, 1, 1), date(2023, 12, 31)),
    dt: datetime = g.DateTime(datetime(2023, 1, 1, 12, 0), datetime(2023, 1, 2, 12, 0)),
):
    """
    Verifies Date and DateTime generators.
    """
    assert isinstance(d, date)
    assert d.year == 2023

    assert isinstance(dt, datetime)
    assert dt.year == 2023


@certify(reliability=0.9, confidence=0.9)
def test_dependent_case_syntax(
    # Type hint says SimpleNamespace (or Any), default is Case generator
    c: SimpleNamespace = g.Case(
        x=g.Integer(1, 10), y=lambda x: g.Integer(x + 1, x + 10)
    )
):
    """
    Verifies that g.Case correctly resolves dependent values using the default syntax.
    """
    assert c.x < c.y
    assert c.y <= c.x + 10


@certify(reliability=0.9, confidence=0.9)
def test_mixing_pytest_fixtures_with_defaults(
    # Standard Pytest fixture (no default)
    tmp_path,
    # Sixma Generator (with default)
    filename: str = g.String(max_len=5, chars="abcde"),
):
    """
    Verifies that we can mix standard pytest fixtures with Sixma default-value generators.
    """
    assert tmp_path.exists()

    # Use both
    f = tmp_path / f"{filename}.txt"
    f.write_text("content")

    assert f.exists()

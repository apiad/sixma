import pytest
from typing import Annotated
from sixma import certify, require, generators
from sixma.core import CertificationError

# Define some handy types for testing
Int100 = Annotated[int, generators.Integer(0, 100)]
AnyInt = Annotated[int, generators.Integer(-100, 100)]


def test_happy_path_certification():
    """
    Verifies that a valid property passes certification.
    """

    @certify(reliability=0.9, confidence=0.9)  # Runs ~21 trials
    def valid_property(a: Int100, b: Int100):
        # Commutative property of addition
        assert a + b == b + a

    # Should run silent and return None
    valid_property() # type: ignore


def test_falsification_detects_bug():
    """
    Verifies that the framework successfully FAILS when a bug exists.
    """

    @certify(reliability=0.9, confidence=0.9)
    def broken_property(a: AnyInt):
        # Bug: Logic fails for negative numbers
        assert a >= 0

    # We expect the framework to raise an AssertionError
    with pytest.raises(AssertionError) as excinfo:
        broken_property() # type: ignore

    # Verify the error message contains helpful info
    error_msg = str(excinfo.value)
    assert "Falsified" in error_msg
    assert "Inputs:" in error_msg


def test_precondition_logic_filtering():
    """
    Verifies that 'require' correctly skips invalid inputs without failing the test.
    """

    @certify(reliability=0.9, confidence=0.9)
    def constrained_property(x: AnyInt):
        # We only want to test even numbers
        require(x % 2 == 0)
        assert x % 2 == 0

    constrained_property() # type: ignore


def test_exhaustion_error():
    """
    Verifies that if we set impossible preconditions, the framework gives up.
    """

    @certify(max_discards=50)  # Set low limit for speed
    def impossible_property(x: Int100):
        # Impossible to satisfy (x cannot be > 200 if generator max is 100)
        require(x > 200)

    with pytest.raises(CertificationError) as excinfo:
        impossible_property() # type: ignore

    assert "Discarded 51 inputs" in str(excinfo.value)


def test_generator_edge_cases_are_hit_first():
    """
    Verifies that our 'Smart' generators actually yield edge cases (0, 1, etc.)
    before random values.
    """

    captured_inputs = []

    # Use a custom generator logic or spy
    # We'll just trust Integer for now and see if it catches 0

    @certify(reliability=0.9, confidence=0.9)
    def spy_test(x: AnyInt):
        captured_inputs.append(x)
        assert True

    spy_test() # type: ignore

    # Assert that critical edge cases were visited
    assert 0 in captured_inputs
    assert 1 in captured_inputs
    assert -1 in captured_inputs
    assert 100 in captured_inputs  # Bounds
    assert -100 in captured_inputs  # Bounds


@certify(reliability=0.9, confidence=0.9)
def test_file_writing(
    # Sixma Param
    n: Int100,
    # Pytest Fixture (Standard argument, no generator annotation)
    tmp_path
):
    """
    Tests that we can write a generated number to a temporary file provided by pytest.
    """
    # Verify tmp_path is real
    assert tmp_path.exists()

    # Use both inputs
    p = tmp_path / f"test_{n}.txt"
    p.write_text(str(n))

    # Assert
    assert p.read_text() == str(n)

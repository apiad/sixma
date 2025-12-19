from datetime import date, timedelta
from sixma import certify, generators as g

# --- Scenario 1: Dependent Integers (The "Slicing" Problem) ---


@certify(reliability=0.99, confidence=0.95)
def test_valid_slice_indices(
    # We generate a struct 'c' where every field depends on the previous ones
    c: g.Case(
        # 1. Driver: Total size of the array
        size=g.Integer(1, 100),
        # 2. Dependent: Start index must be valid for this size
        start=lambda size: g.Integer(0, size - 1),
        # 3. Dependent: End index must be strictly greater than start
        end=lambda start, size: g.Integer(start + 1, size),
    ), # type: ignore
):
    """
    Verifies that we can generate (start, end) pairs that are always valid
    slices for a given size, without using rejection sampling.
    """
    # 1. Verify Structure (Dot notation)
    assert isinstance(c.size, int)

    # 2. Verify Logic
    # If this fails, the generator logic itself is broken
    assert 0 <= c.start < c.end <= c.size

    # 3. Practical Usage
    # We can now safely use these indices without fear of IndexError
    data = list(range(c.size))
    chunk = data[c.start : c.end]

    assert len(chunk) == c.end - c.start
    assert len(chunk) > 0


# --- Scenario 2: Mixed Types (Dates + Integers) ---


@certify(reliability=0.99, confidence=0.9)
def test_project_timeline(
    plan: g.Case(
        # 1. Driver: Project starts sometime in 2024
        start_date=g.Date(date(2024, 1, 1), date(2024, 12, 31)),
        # 2. Dependent: Duration depends on nothing (independent),
        #    but is used in the next step.
        duration_days=g.Integer(1, 30),
        # 3. Dependent: Deadline must be calculated from start + duration
        #    Note: We receive a 'date' object and an 'int' here.
        deadline=lambda start_date, duration_days: g.Date(
            start_date + timedelta(days=duration_days),
            start_date + timedelta(days=duration_days + 14),  # 2 week buffer
        ),
    ), # type: ignore
):
    """
    Verifies complex object interaction: calculating dates based on
    integers generated in the same Case.
    """
    # Sanity checks
    assert plan.start_date.year == 2024

    # Verify the dependent logic held true
    min_deadline = plan.start_date + timedelta(days=plan.duration_days)

    assert plan.deadline >= min_deadline
    assert plan.deadline <= min_deadline + timedelta(days=14)

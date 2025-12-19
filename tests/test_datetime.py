from datetime import date, datetime, timedelta
from typing import Annotated
from sixma import certify, generators as g

# --- 1. Date Testing: The Leap Year Trap ---

# A range covering a leap year (2024) and non-leap years
DateRange = Annotated[date, g.Date(date(2023, 1, 1), date(2025, 12, 31))]


@certify(reliability=0.99, confidence=0.95)
def test_leap_year_properties(d: DateRange):
    """
    Verifies that if it is Feb 29, the year must be a leap year.
    """
    if d.month == 2 and d.day == 29:
        # Standard leap year logic
        is_leap = (d.year % 4 == 0 and d.year % 100 != 0) or (d.year % 400 == 0)
        assert is_leap is True


@certify(reliability=0.99, confidence=0.95)
def test_next_day_logic(d: DateRange):
    """
    Verifies that adding 1 day always results in a date strictly in the future.
    """
    next_day = d + timedelta(days=1)
    assert next_day > d
    assert (next_day - d).days == 1


# --- 2. DateTime Testing: ISO 8601 Roundtrip ---

# A specific time window (e.g., server uptime logs)
LogWindow = Annotated[
    datetime,
    g.DateTime(
        datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 2, 0, 0, 0)  # One day window
    ),
]


@certify(reliability=0.99, confidence=0.95)
def test_iso_format_roundtrip(dt: LogWindow):
    """
    Verifies that datetime -> string -> datetime preserves equality.
    """
    iso_str = dt.isoformat()
    parsed_dt = datetime.fromisoformat(iso_str)

    # Note: fromisoformat might lose precision if microsecond logic varies,
    # so we often test equality.
    assert parsed_dt == dt

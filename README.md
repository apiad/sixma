Here is the completely updated `README.md`. I have restructured the **Quick Start** section to explicitly compare the three syntax styles, clearly labeling them as **Recommended (Production)**, **Strict (Annotated)**, and **Prototyping (Shortcut)**.

```markdown
# Sixma

![PyPI - Version](https://img.shields.io/pypi/v/sixma)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/sixma)
![Github - Open Issues](https://img.shields.io/github/issues-raw/apiad/sixma)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/sixma)
![Github - Commits](https://img.shields.io/github/commit-activity/m/apiad/sixma)

**Probabilistic Correctness & Logical Falsification for Python.**

> "Stop writing unit tests. Start certifying reliability."

Sixma is a testing framework that replaces manual test cases with **Generative Spaces** and **Statistical Certification**. Instead of checking if `f(2) == 4`, you define the invariant `f(x) == x^2` and Sixma proves it holds true with a specific **Reliability** and **Confidence Level**.

It uses the **Zero-Failure Reliability** model to dynamically calculate the required number of random trials ($N$) to certify your system is bug-free.
## 📦 Installation

```bash
uv add sixma
# or
pip install sixma

```

## 🚀 Quick Start: The 3 Ways to Write Tests

Sixma is flexible. Choose the syntax that fits your project's strictness level.

### 1. The Production Way (Recommended) ⭐

Use standard Python type hints combined with **Default Values**.

* **Pros:** 100% compatible with `mypy`, `pyright`, and IDE autocomplete.
* **Cons:** Slightly more verbose.

```python
from sixma import certify, generators as g

@certify(reliability=0.999, confidence=0.95)
def test_math(
    # Mypy sees 'int'. Sixma sees the generator in the default value.
    x: int = g.Integer(0, 100),
    y: int = g.Integer(0, 100)
):
    assert x + y == y + x

```

### 2. The Prototyping Way (Shortcut) ⚡

Use the generator instance directly as the type hint.

* **Pros:** Fastest to write. Very clean to read.
* **Cons:** Static type checkers (mypy) will complain that "Integer is not a type".

```python
@certify
def test_math(
    # Fast to write, but upsets strict type checkers
    x: g.Integer(0, 100),
    # or even as default value
    y = g.Integer(0, 100),
):
    assert x + y == y + x

```

### 3. The Strict Way (Annotated) 🧐

Use Python's standard `typing.Annotated`.

* **Pros:** The "Academic" standard for metadata. Mypy compliant.
* **Cons:** Very verbose.

```python
from typing import Annotated

@certify
def test_math(
    x: Annotated[int, g.Integer(0, 100)],
    y: Annotated[int, g.Integer(0, 100)]
):
    assert x + y == y + x

```
## 🧠 The Philosophy

Standard property-based testing runs an arbitrary number of tests (e.g., 100). Sixma inverts this: **You tell the framework how confident you want to be.**

The number of trials  is calculated dynamically using the Zero-Failure Testing formula:

$$
N = \left\lceil \frac{\ln(1 - C)}{\ln(R)} \right\rceil
$$

| Reliability | Confidence | Trials Required | Use Case                |
| ----------- | ---------- | --------------- | ----------------------- |
| 0.90        | 0.95       | 29              | MVP / Quick Smoke Tests |
| 0.99        | 0.99       | 459             | Standard Business Logic |
| 0.999       | 0.99       | 4,603           | Core Algorithms         |
| 0.9999      | 0.999      | 69,075          | Critical Infrastructure |

* **Reliability:** The probability that the code will NOT fail on a random input.
* **Confidence:** The probability that our estimation of  is correct.

## 🛠 Features

### 1. Smart Generators (Edge Cases First)

Sixma generators are **finite iterators** first. They always yield critical edge cases (0, -1, empty strings, boundaries, leap years) before switching to random sampling.

```python
# Yields: 0, 10, 1, -1, 5, 8, ...
x: int = g.Integer(0, 10)

```

### 2. Dependent Variables (`g.Case`)

Define inputs that depend on each other without wasteful rejection sampling.

```python
@certify
def test_slicing(
    # 'case' generates a namespace where fields depend on previous ones
    case: SimpleNamespace = g.Case(
        # 1. Independent Variable
        size = g.Integer(1, 100),

        # 2. Dependent: start must be within size
        start = lambda size: g.Integer(0, size - 1),

        # 3. Dependent: end must be > start
        end = lambda start, size: g.Integer(start + 1, size)
    )
):
    # Inputs are guaranteed to be valid!
    # Mypy Tip: Use SimpleNamespace or a custom class for type hints.
    data = list(range(case.size))
    chunk = data[case.start : case.end]

    assert len(chunk) == case.end - case.start

```

### 3. Time Travel (Temporal Testing)

Generate valid dates, times, and windows easily. Handles leap years automatically.

```python
from datetime import date

# Define a Business Quarter
Q1_2024 = g.Date(date(2024, 1, 1), date(2024, 3, 31))

@certify
def test_quarterly_report(day: date = Q1_2024):
    assert day.year == 2024
    assert 1 <= day.month <= 3

```

### 4. Reproducibility (Seeding)

Statistical tests must be reproducible. If a test fails, Sixma prints the **Random Seed** used in the logs.

**Output on Failure:**

```text
❌ Falsified at trial 412!
   Seed: 84920174 (Set SIXMA_SEED=84920174 to reproduce)
   Inputs: {'x': -5}
   Error: assert -5 > 0

```

**Reproduce it locally:**

```bash
SIXMA_SEED=84920174 pytest tests/test_my_logic.py

```

### 5. Auto-Shrinking (Debuggability)

When a test fails on a complex input (e.g., a list of 50 items), Sixma automatically re-runs the test with the **Simplest Case** (e.g., empty list) to see if the bug persists.

```text
❌ Falsified at trial 55!
   Inputs: {'items': [23, 99, ... 48 more]}
   📉 Minimal Counter-Example: {'items': []}

```
## 📚 API Reference

### `@certify(reliability, confidence, max_discards)`

The main decorator.

* `reliability`: Target probability of success (0.0 - 1.0).
* `confidence`: Statistical significance level (0.0 - 1.0).
* `max_discards`: Safety valve for infinite loops in `require()`.

### Generators (`sixma.generators`)

All generators are available as factory functions compatible with Mypy.

* **Primitives:** `Integer`, `Float`, `Bool`, `String`
* **Combinators:**
  * `List(gen, min_len, max_len)`
  * `Dict(key=gen, ...)`
  * `Object(Cls, field=gen, ...)`
* **Logic:**
  * `Case(field=gen, dependent_field=lambda prev: gen)`
* **Temporal:**
  * `Date(start, end)`
  * `DateTime(start, end)`


## 📄 License

MIT License.

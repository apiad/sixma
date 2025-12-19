# Sixma

![PyPI - Version](https://img.shields.io/pypi/v/sixma)
![PyPi - Python Version](https://img.shields.io/pypi/pyversions/sixma)
![Github - Open Issues](https://img.shields.io/github/issues-raw/apiad/sixma)
![PyPi - Downloads (Monthly)](https://img.shields.io/pypi/dm/sixma)
![Github - Commits](https://img.shields.io/github/commit-activity/m/apiad/sixma)

**Probabilistic Correctness & Logical Falsification for Python.**

> "Stop writing unit tests. Start certifying reliability."

Sixma is a testing framework that replaces manual test cases with **Generative Spaces** and **Statistical Certification**. Instead of checking if `f(2) == 4`, you define the invariant `f(x) == x^2` and Sixma proves it holds true with a specific **Reliability** and **Confidence Level**.

It is built on the **Zero-Failure Reliability** model: calculating exactly how many random trials are required to certify that a system is bug-free up to a certain probability threshold.

## 📦 Installation

```bash
uv add sixma
# or
pip install sixma
```

## 🚀 Quick Start

Write your tests as standard Python functions using the `sixma.generators` as type hints.

```python
from sixma import certify, require, generators as g

# 1. Certify logic
@certify(reliability=0.999, confidence=0.95)
def test_drinking_age_logic(age: g.Integer(0, 120)):
    """
    Verifies legal drinking age logic.
    Target: 0.1% failure rate with 95% confidence (~2993 trials).
    """
    # Preconditions (Rejection Sampling)
    # If this is False, the input is discarded and regenerated.
    require(age >= 18)

    # Execution
    is_allowed = check_id(age)

    # Postconditions (Falsification)
    # If this fails ONCE, the certification fails immediately.
    assert is_allowed is True
```

Run it with standard `pytest`:

```bash
$ pytest -s

[Sixma] Target: 2993 successes (R=0.999, C=0.95)
[Sixma] Certified ✔️  (2993 passed, 41 discarded)
PASSED
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

* **Reliability ():** The probability that the code will NOT fail on a random input.
* **Confidence ():** The probability that our estimation of  is correct.

## 🛠 Features

### 1. Smart Generators

Sixma generators are **finite iterators** first, and **infinite streams** second. They always yield edge cases (0, -1, empty strings, boundaries) before switching to random sampling.

```python
# Will yield: 0, 10, 1, -1, 5, 8, ...
def test_integers(x: g.Integer(0, 10)): ...

# Will yield: "", "a", "   ", "xyz", ...
def test_strings(s: g.String(max_len=3)): ...
```

### 2. Clean Syntax

You don't need `typing.Annotated`. Just use the generator instance directly as the type hint.

```python
@certify
def test_math(
    x: g.Integer(0, 100),
    y: g.Float(0.0, 1.0)
):
    assert x * y <= 100.0
```

### 3. Dependent Variables (`g.Case`)

Often, test inputs depend on each other (e.g., `start < end` or `deadline > created_at`). Instead of using `require()`—which wastes CPU cycles discarding invalid random samples—use `g.Case` to define a **Generative Causal Chain**.

```python
@certify
def test_slicing(
    # 'case' generates a namespace where fields depend on previous ones
    case: g.Case(
        # 1. Independent Variable
        size = g.Integer(1, 100),

        # 2. Dependent: start must be within size
        start = lambda size: g.Integer(0, size - 1),

        # 3. Dependent: end must be > start
        end = lambda start, size: g.Integer(start + 1, size)
    )
):
    # Inputs are guaranteed to be valid! No require() needed.
    data = list(range(case.size))
    chunk = data[case.start : case.end]

    assert len(chunk) == case.end - case.start
```

### 4. Complex Data Structures

Model complex domains easily with `List`, `Dict`, and `Object`.

```python
@dataclass
class Packet:
    id: int
    payload: str

# Generates fully populated class instances
def test_network(
    pkts: g.List(
        g.Object(
            Packet,
            id=g.Integer(1000, 9999),
            payload=g.String(max_len=255)
        ),
        min_len=1, max_len=5
    )
):
    assert len(pkts) >= 1
```

### 5. Time Travel (Temporal Testing)

Generate valid dates, times, and windows easily.

```python
from datetime import date, timedelta

# Define a Business Quarter
Q1_2024 = g.Date(date(2024, 1, 1), date(2024, 3, 31))

@certify
def test_quarterly_report(day: Q1_2024):
    assert day.year == 2024
    assert 1 <= day.month <= 3
```

### 6. Reproducibility (Seeding)

Statistical tests must be reproducible. If a test fails, Sixma prints the **Random Seed** used.

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
## 📚 API Reference

### `@certify(reliability, confidence, max_discards)`

The main decorator.

* `reliability`: Target probability of success (0.0 - 1.0).
* `confidence`: Statistical significance level (0.0 - 1.0).
* `max_discards`: Safety valve for infinite loops in `require()`.

### `require(condition)`

Used for **Preconditions**.

* If `True`: Continues execution.
* If `False`: Aborts the current trial, discards inputs, and samples again.

### Generators (`sixma.generators`)

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

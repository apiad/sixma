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

---

## 📦 Installation

```bash
uv add sixma
# or
pip install sixma
```

## 🚀 Quick Start

Write your tests as standard Python functions, but use `Annotated` to define input domains and `@certify` to define rigor.

```python
from typing import Annotated
from sixma import certify, require, generators as gen

# 1. Define reusable domains
UserAge = Annotated[int, gen.Integer(0, 120)]

# 2. Certify logic
@certify(reliability=0.999, confidence=0.95)
def test_drinking_age_logic(age: UserAge):
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

### Smart Generators

Sixma generators are **finite iterators** first, and **infinite streams** second. They always yield edge cases (0, -1, empty strings, boundaries) before switching to random sampling.

```python
# Will yield: 0, 10, 1, -1, 5, 8, ...
gen.Integer(0, 10)

# Will yield: "", "a", "   ", "xyz", ...
gen.String(max_len=3)

```

### Complex Combinators

Model complex domains easily with `List`, `Dict`, and `Object`.

```python
@dataclass
class Packet:
    id: int
    payload: str

# Generates fully populated class instances
gen.Object(
    Packet,
    id=gen.Integer(1000, 9999),
    payload=gen.String(max_len=255)
)

```

### Pytest Integration

Sixma patches the function signature so `pytest` fixtures work seamlessly alongside generated inputs.

```python
@certify(reliability=0.99)
def test_database_write(
    user: Annotated[User, UserGen], # Injected by Sixma
    db_session                      # Injected by Pytest
):
    db_session.add(user)
    assert user.id in db_session

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

## 📄 License

MIT License.

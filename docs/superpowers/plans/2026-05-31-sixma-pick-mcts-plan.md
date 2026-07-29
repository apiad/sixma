# Sixma v1 (pick-based test bodies) Implementation Plan

**Status: complete — shipped as v0.5.0 (`32f59c9`, 2026-05-31).** All 5 tasks,
8/8 acceptance criteria, 37 tests passing, published to PyPI and tagged on
GitHub. Deferred axes live in §5 of the design doc.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `ctx` parameter to `@certify` that lets test bodies pull random values mid-execution via `ctx.pick(...)`, `ctx.range(...)`, and `ctx.discard()`, certifying invariants over the body's induced path distribution under uniform sampling.

**Architecture:** A new internal triple — `Trial` (per-trial state), `Sampler` interface (with `UniformSampler` v1), and `PickContext` (the user-facing facade) — is orchestrated by the existing `certify` decorator. Parameter inspection at decoration time identifies at most one non-generator parameter and binds it to a freshly constructed `PickContext` per trial. The existing default-value / `Annotated` / raw-instance generator paths are untouched, so all 23 existing tests pass verbatim. Failures from `ctx`-using trials emit a multi-line `Trace:` block instead of the single-line `Inputs:` dict.

**Tech Stack:** Python 3.12+, `random.Random`, `inspect`, `functools`, uv, pytest, ruff.

---

## File Structure

- Modify: `src/sixma/core.py` — add `_Discard` sentinel, `Trial`, `Sampler`/`UniformSampler`, `PickContext`; rework `certify` decorator to detect a `ctx` slot and inject a `PickContext` per trial; new `max_picks_per_trial` kwarg; new trace-shaped failure message.
- Modify: `src/sixma/__init__.py` — re-export `PickContext` (public type, useful for users wanting to type-hint `ctx: PickContext`).
- Create: `tests/test_picks.py` — the vertical-slice Stack/LIFO test + acceptance suite covering criteria 2-8 from spec §4.
- Modify: `README.md` — add a "4. The Pick-Based Way" section + document `max_picks_per_trial` in API reference.
- Modify: `pyproject.toml` — bump version to `0.5.0` (release task only, not feature task).

## Self-Review Notes

Done after writing. Key symbol-existence checks:

- `BaseGenerator` in `src/sixma/generators.py:28` ✓
- `certify` in `src/sixma/core.py:29` ✓
- `PreconditionError` / `CertificationError` in `src/sixma/core.py:19,22` ✓
- `require` in `src/sixma/core.py:25` ✓
- `get_type_hints` import at `src/sixma/core.py:7` ✓
- `_Integer` and the smart_sample machinery stay untouched ✓
- The decorator currently strips sixma params via `wrapper.__signature__ = sig.replace(parameters=...)`. The `ctx` parameter must also be stripped. ✓ — addressed in Task 3.

---

## Task 1: Add Sampler + Trial + PickContext primitives

**Files:**
- Modify: `src/sixma/core.py` (add new classes before the `certify` function)
- Test: `tests/test_picks.py` (create)

- [x] **Step 1: Write the failing unit tests for PickContext primitives**

Create `tests/test_picks.py` with:

```python
"""Unit tests for PickContext / Sampler / Trial primitives + decorator integration."""
import random
import pytest

from sixma import certify, generators as g
from sixma.core import (
    CertificationError,
    PickContext,
    Trial,
    UniformSampler,
)


def test_pick_returns_one_of_options():
    rng = random.Random(0)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    chosen = ctx.pick("a", "b", "c")
    assert chosen in {"a", "b", "c"}


def test_pick_zero_options_raises():
    rng = random.Random(0)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    with pytest.raises(ValueError):
        ctx.pick()


def test_range_returns_integer_in_bounds():
    rng = random.Random(0)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    for _ in range(50):
        v = ctx.range(-5, 5)
        assert -5 <= v <= 5


def test_pick_records_trace_with_auto_labels():
    rng = random.Random(42)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    ctx.pick("a", "b")
    ctx.range(0, 10)
    ctx.pick(1, 2, 3, label="explicit")
    labels = [label for label, _ in trial.picks]
    assert labels == ["pick_0", "pick_1", "explicit"]


def test_discard_raises_internal_sentinel():
    from sixma.core import _Discard
    rng = random.Random(0)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    with pytest.raises(_Discard):
        ctx.discard()


def test_pick_counter_increments_per_call():
    rng = random.Random(0)
    trial = Trial(rng=rng)
    ctx = PickContext(trial=trial, sampler=UniformSampler())
    ctx.pick(1, 2)
    ctx.range(0, 5)
    assert trial.pick_count == 2
```

- [x] **Step 2: Run tests; verify they fail**

Run: `uv run pytest tests/test_picks.py -v`
Expected: ImportError / AttributeError on `PickContext`, `Trial`, `UniformSampler`, `_Discard`.

- [x] **Step 3: Add the primitive classes to `src/sixma/core.py`**

Insert these definitions immediately after the `CertificationError` class (around line 24) and before `require`:

```python
class _Discard(Exception):
    """Internal sentinel raised by ctx.discard() — caught by the trial loop."""
    pass


class _TrialTooDeep(Exception):
    """Internal sentinel raised when a trial exceeds max_picks_per_trial."""
    pass


class Trial:
    """Per-trial state. Not user-facing."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.picks: list[tuple[str, object]] = []
        self.pick_count: int = 0


class UniformSampler:
    """Uniform random sampler. The v1 default."""

    def choose(self, options, trial: Trial):
        return trial.rng.choice(options)

    def randint(self, low: int, high: int, trial: Trial) -> int:
        return trial.rng.randint(low, high)


class PickContext:
    """Test-body-facing facade. Records each decision into `trial.picks` and
    delegates the random choice to the sampler.
    """

    def __init__(self, trial: Trial, sampler, max_picks: int = 1000):
        self._trial = trial
        self._sampler = sampler
        self._max_picks = max_picks

    @property
    def trace(self) -> list[tuple[str, object]]:
        return list(self._trial.picks)

    def _check_budget(self):
        if self._trial.pick_count >= self._max_picks:
            raise _TrialTooDeep()
        self._trial.pick_count += 1

    def _label(self, label):
        if label is None:
            return f"pick_{len(self._trial.picks)}"
        return label

    def pick(self, *options, label=None):
        if not options:
            raise ValueError("ctx.pick requires at least one option")
        self._check_budget()
        chosen = self._sampler.choose(list(options), self._trial)
        self._trial.picks.append((self._label(label), chosen))
        return chosen

    def range(self, low: int, high: int, label=None) -> int:
        self._check_budget()
        chosen = self._sampler.randint(low, high, self._trial)
        self._trial.picks.append((self._label(label), chosen))
        return chosen

    def discard(self):
        raise _Discard()
```

- [x] **Step 4: Run tests; verify they pass**

Run: `uv run pytest tests/test_picks.py -v`
Expected: 6 passed.

- [x] **Step 5: Run full suite + ruff**

Run: `uv run ruff check . && uv run pytest -q`
Expected: 0 lint errors, all tests pass (23 existing + 6 new = 29).

- [x] **Step 6: Commit**

```bash
git add src/sixma/core.py tests/test_picks.py
git commit -m "feat(core): PickContext + Sampler seam for pick-based test bodies"
```

---

## Task 2: Wire `ctx` parameter into the `@certify` decorator

**Files:**
- Modify: `src/sixma/core.py` — parameter discovery + trial loop changes
- Modify: `src/sixma/__init__.py` — export `PickContext`
- Test: `tests/test_picks.py` — integration tests

- [x] **Step 1: Add integration tests at the end of `tests/test_picks.py`**

Append:

```python
def test_pick_only_test_certifies_trivial_invariant():
    """Acceptance criterion 5: ctx-only test with trivially-true invariant certifies."""
    @certify(reliability=0.9, confidence=0.9)
    def trivially_true(ctx):
        v = ctx.pick(1, 2, 3)
        assert v in (1, 2, 3)

    trivially_true()  # type: ignore


def test_mixed_generator_and_ctx():
    """Acceptance criterion 6: default-value generator + ctx together."""
    @certify(reliability=0.9, confidence=0.9)
    def mixed(x: int = g.Integer(0, 10), ctx=None):
        delta = ctx.range(-5, 5)
        result = x + delta
        assert -5 <= result <= 15

    mixed()  # type: ignore


def test_two_ctx_params_raises_at_decoration():
    with pytest.raises(ValueError, match="at most one"):
        @certify(reliability=0.9, confidence=0.9)
        def two_ctx(ctx1, ctx2):
            pass


def test_discard_via_ctx_counts_as_precondition():
    """ctx.discard() funnels through the same machinery as require()."""
    @certify(reliability=0.9, confidence=0.9)
    def half_discarded(ctx):
        x = ctx.range(0, 1)
        if x == 0:
            ctx.discard()
        assert x == 1

    half_discarded()  # type: ignore
```

- [x] **Step 2: Run tests; verify they fail**

Run: `uv run pytest tests/test_picks.py::test_pick_only_test_certifies_trivial_invariant -v`
Expected: TypeError or similar — the decorator doesn't yet know what to do with `ctx`.

- [x] **Step 3: Modify the `certify` decorator signature in `src/sixma/core.py`**

Replace the current signature:

```python
def certify(
    reliability: float = 0.999, confidence: float = 0.95, max_discards: int = 10000
):
```

with:

```python
def certify(
    reliability: float = 0.999,
    confidence: float = 0.95,
    max_discards: int = 10000,
    max_picks_per_trial: int = 1000,
):
```

- [x] **Step 4: Modify parameter-discovery block in the decorator**

Right after the existing Strategy 1 + Strategy 2 blocks (around line 78), add Strategy 3 — detect at most one `ctx`-style parameter:

```python
# Strategy 3: Identify the ctx (PickContext) parameter — at most one.
ctx_param_name = None
for name, param in sig.parameters.items():
    if name in sixma_param_names:
        continue
    # Skip pytest fixtures (heuristic: no default, no relevant annotation)
    # The ctx slot is: any non-generator-bound, non-fixture-looking param named "ctx"
    # OR explicitly typed as PickContext.
    type_hint = test_func.__annotations__.get(name)
    is_ctx = (
        name == "ctx"
        or type_hint is PickContext
        or (isinstance(type_hint, type) and issubclass(type_hint, PickContext))
    )
    if is_ctx:
        if ctx_param_name is not None:
            raise ValueError(
                f"at most one ctx parameter allowed; "
                f"found '{ctx_param_name}' and '{name}'"
            )
        ctx_param_name = name
        sixma_param_names.add(name)
```

- [x] **Step 5: Modify the trial loop to build & inject `PickContext`**

Replace the trial loop body (roughly lines 116-160) with the version below. The behavioral diff: build a fresh `Trial` + `PickContext` per trial when `ctx_param_name` is set; catch `_Discard` and `_TrialTooDeep` as discards; emit a `Trace:` failure message for ctx-using trials.

```python
while successes < required_successes:
    if discards > max_discards:
        raise CertificationError(f"Discarded {discards} inputs.")

    # Generate generator-bound kwargs
    generated_kwargs = {}
    for name, stream in active_streams.items():
        try:
            generated_kwargs[name] = next(stream)
        except StopIteration:
            raise RuntimeError(f"Generator for '{name}' exhausted.")

    # Build a PickContext for this trial if the body wants one
    trial = None
    ctx_obj = None
    if ctx_param_name is not None:
        trial = Trial(rng=rng)
        ctx_obj = PickContext(
            trial=trial,
            sampler=UniformSampler(),
            max_picks=max_picks_per_trial,
        )
        generated_kwargs[ctx_param_name] = ctx_obj

    final_kwargs = {**fixture_kwargs, **generated_kwargs}

    try:
        test_func(**final_kwargs)
        successes += 1
    except PreconditionError:
        discards += 1
        continue
    except _Discard:
        discards += 1
        continue
    except _TrialTooDeep:
        discards += 1
        continue
    except AssertionError as e:
        if ctx_param_name is not None and trial is not None:
            # Pick-based failure: render trace
            trace_lines = []
            picks = trial.picks
            if len(picks) > 20:
                shown = picks[:10] + [("...", f"({len(picks) - 15} picks elided)")] + picks[-5:]
            else:
                shown = picks
            for label, value in shown:
                trace_lines.append(f"     {label} = {value!r}")
            trace_block = "\n".join(trace_lines) if trace_lines else "     (no picks)"
            error_msg = (
                f"❌ Falsified at trial {successes + 1}!\n"
                f"   Seed: {current_seed} (Set SIXMA_SEED={current_seed} to reproduce)\n"
                f"   Trace:\n{trace_block}\n"
                f"   Error: {e}"
            )
            logger.error(error_msg)
            raise AssertionError(error_msg) from e
        else:
            # Legacy-path failure: keep existing Inputs+shrinking shape
            minimal_msg = ""
            try:
                min_kwargs = {}
                for name, bp in generator_blueprints.items():
                    gen_inst = bp() if isinstance(bp, type) else bp
                    min_kwargs[name] = next(iter(gen_inst))

                final_min_kwargs = {**fixture_kwargs, **min_kwargs}
                test_func(**final_min_kwargs)
            except AssertionError:
                minimal_msg = f"\n   📉 Minimal Counter-Example: {min_kwargs}"
            except Exception:
                pass

            error_msg = (
                f"❌ Falsified at trial {successes + 1}!\n"
                f"   Seed: {current_seed} (Set SIXMA_SEED={current_seed} to reproduce)\n"
                f"   Inputs: { {k: v for k, v in generated_kwargs.items() if k != ctx_param_name} }"
                f"{minimal_msg}\n"
                f"   Error: {e}"
            )
            logger.error(error_msg)
            raise AssertionError(error_msg) from e
```

Note the `{k: v for k, v in ... if k != ctx_param_name}` filter — in the mixed path the `ctx` should never appear in `Inputs:`, but we never enter this branch when `ctx_param_name is not None` anyway. The expression is safe with `ctx_param_name=None` because the dict comp short-circuits.

- [x] **Step 6: Re-export `PickContext` from `src/sixma/__init__.py`**

Update the file to:

```python
from importlib.metadata import PackageNotFoundError, version

from .core import certify, require, PickContext

try:
    __version__ = version("sixma")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


__all__ = ["certify", "require", "PickContext"]
```

- [x] **Step 7: Run targeted tests; verify they pass**

Run: `uv run pytest tests/test_picks.py -v`
Expected: 10 passed.

- [x] **Step 8: Run full suite + ruff**

Run: `uv run ruff check . && uv run pytest -q`
Expected: 0 lint errors, 33 tests pass (23 existing + 10 new).

- [x] **Step 9: Commit**

```bash
git add src/sixma/core.py src/sixma/__init__.py tests/test_picks.py
git commit -m "feat(core): wire ctx parameter into certify with max_picks_per_trial safety valve"
```

---

## Task 3: Vertical-slice Stack/LIFO test + safety-valve acceptance tests

**Files:**
- Modify: `tests/test_picks.py` — add Stack test + safety-valve tests

- [x] **Step 1: Add the Stack vertical-slice test + safety-valve tests**

Append to `tests/test_picks.py`:

```python
class Stack:
    """Stack with an injected LIFO bug at len == 3."""
    def __init__(self):
        self._items = []
    def push(self, x):
        self._items.append(x)
    def pop(self):
        if len(self._items) == 3:
            return self._items.pop(0)  # Bug
        return self._items.pop()
    def size(self):
        return len(self._items)


def test_vertical_slice_stack_falsifies():
    """Acceptance criterion 2 + 3: the vertical-slice falsifies and reports a trace."""
    @certify(reliability=0.99, confidence=0.95, max_picks_per_trial=100)
    def stack_lifo(ctx):
        stack = Stack()
        reference = []
        while ctx.pick("op", "stop", label="loop") == "op":
            if ctx.pick("push", "pop", label="action") == "push":
                x = ctx.range(0, 100, label="value")
                stack.push(x)
                reference.append(x)
            else:
                if not reference:
                    ctx.discard()
                assert stack.pop() == reference.pop()

    with pytest.raises(AssertionError) as excinfo:
        stack_lifo()  # type: ignore
    msg = str(excinfo.value)
    assert "Falsified" in msg
    assert "Trace:" in msg
    assert "Seed:" in msg
    assert "SIXMA_SEED=" in msg


def test_seed_reproduces_falsification(monkeypatch):
    """Acceptance criterion 4: SIXMA_SEED reproduces the exact failing trial."""
    # Round 1: capture the seed from a failing run
    @certify(reliability=0.99, confidence=0.95, max_picks_per_trial=100)
    def stack_lifo(ctx):
        stack = Stack()
        reference = []
        while ctx.pick("op", "stop", label="loop") == "op":
            if ctx.pick("push", "pop", label="action") == "push":
                x = ctx.range(0, 100, label="value")
                stack.push(x)
                reference.append(x)
            else:
                if not reference:
                    ctx.discard()
                assert stack.pop() == reference.pop()

    with pytest.raises(AssertionError) as first:
        stack_lifo()  # type: ignore
    msg1 = str(first.value)
    # Extract seed
    import re
    m = re.search(r"SIXMA_SEED=(\d+)", msg1)
    assert m, f"seed not found in: {msg1}"
    seed = m.group(1)

    # Round 2: same seed should give the same trace
    monkeypatch.setenv("SIXMA_SEED", seed)
    with pytest.raises(AssertionError) as second:
        stack_lifo()  # type: ignore
    msg2 = str(second.value)
    # The trace block should match (everything between "Trace:" and "Error:")
    def trace_of(m):
        i = m.index("Trace:")
        j = m.index("Error:")
        return m[i:j]
    assert trace_of(msg1) == trace_of(msg2)


def test_divergent_body_discards_via_max_picks_per_trial():
    """Acceptance criterion 7: while-True picks doesn't hang; discards instead."""
    @certify(
        reliability=0.5,
        confidence=0.5,
        max_picks_per_trial=10,
        max_discards=5,
    )
    def runaway(ctx):
        while True:
            ctx.pick(1, 2)
        assert False  # unreachable

    with pytest.raises(CertificationError):
        runaway()  # type: ignore


def test_max_picks_exhaustion_then_certification_error():
    """Acceptance criterion 8: pick budget exhaustion bumps the discard counter."""
    @certify(
        reliability=0.5,
        confidence=0.5,
        max_picks_per_trial=3,
        max_discards=2,
    )
    def too_deep(ctx):
        for _ in range(100):
            ctx.pick(1, 2)
        assert True

    with pytest.raises(CertificationError) as excinfo:
        too_deep()  # type: ignore
    assert "Discarded" in str(excinfo.value)
```

- [x] **Step 2: Run tests; verify they pass**

Run: `uv run pytest tests/test_picks.py -v`
Expected: 14 passed.

- [x] **Step 3: Run full suite + ruff**

Run: `uv run ruff check . && uv run pytest -q`
Expected: 0 lint errors, 37 tests pass.

- [x] **Step 4: Commit**

```bash
git add tests/test_picks.py
git commit -m "test(picks): vertical-slice Stack/LIFO + safety-valve acceptance suite"
```

---

## Task 4: README update

**Files:**
- Modify: `README.md` — add "4. The Pick-Based Way" + document `max_picks_per_trial`

- [x] **Step 1: Insert a "4. The Pick-Based Way" section after section "3. The Strict Way"**

Find the line at the end of section 3 (just before `## 🧠 The Philosophy`) and insert:

```markdown
### 4. The Pick-Based Way (Stateful & Coupled) 🎲

For tests where inputs are coupled to control flow — state machines, op sequences, anything where "the next value depends on what we just did" — declare a `ctx` parameter. The framework injects a `PickContext` the body pulls values from on demand.

```python
from sixma import certify

@certify(reliability=0.99, confidence=0.95, max_picks_per_trial=100)
def test_stack_lifo(ctx):
    stack, reference = [], []
    while ctx.pick("op", "stop") == "op":
        if ctx.pick("push", "pop") == "push":
            x = ctx.range(0, 100)
            stack.append(x); reference.append(x)
        else:
            if not reference: ctx.discard()
            assert stack.pop() == reference.pop()
```

* `ctx.pick(*options, label=None)` — uniformly random choice from the options.
* `ctx.range(low, high, label=None)` — integer in `[low, high]` inclusive.
* `ctx.discard()` — equivalent to `require(False)`; the trial doesn't count.

**Semantic shift**: under picks, `reliability` characterizes the fraction of the *test body's path distribution* that passes — not an external input space. Bugs hiding behind unlikely pick sequences are exponentially under-sampled; restructure the body or use `ctx.discard()` to bias the distribution toward interesting paths.

**Mixing**: a body may declare both default-value generators and `ctx`:

```python
@certify
def test_mixed(x: int = g.Integer(0, 100), ctx=None):
    while ctx.pick(True, False):
        x += ctx.range(-10, 10)
    assert -10000 < x < 10000
```

Failure traces show the ordered list of picks plus the seed for reproduction:

```text
❌ Falsified at trial 412!
   Seed: 84920174 (Set SIXMA_SEED=84920174 to reproduce)
   Trace:
     loop = 'op'
     action = 'push'
     value = 42
     loop = 'op'
     action = 'pop'
   Error: AssertionError
```
```

(Note: in the actual file, the ` ``` ` block above is inline markdown — preserve the nested code fences using the existing four-space indent convention if needed.)

- [x] **Step 2: Update the API Reference section**

Find:

```markdown
### `@certify(reliability, confidence, max_discards)`

The main decorator.

* `reliability`: Target probability of success (0.0 - 1.0).
* `confidence`: Statistical significance level (0.0 - 1.0).
* `max_discards`: Safety valve for infinite loops in `require()`.
```

Replace with:

```markdown
### `@certify(reliability, confidence, max_discards, max_picks_per_trial)`

The main decorator.

* `reliability`: Target probability of success (0.0 - 1.0).
* `confidence`: Statistical significance level (0.0 - 1.0).
* `max_discards`: Safety valve — caps the number of discarded trials (`require()` failures and `ctx.discard()` calls combined).
* `max_picks_per_trial`: Safety valve for divergent test bodies that loop on `ctx.pick(...)`. Default 1000. Exceeding this discards the trial.
```

- [x] **Step 3: Visually verify rendering**

Run: `head -200 README.md` and confirm the new section reads cleanly.

- [x] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document the ctx-based test-body syntax"
git push origin main
```

---

## Task 5: Release v0.5.0

Follow `know-how/releasing.md` step-by-step. Summary below — refer to the know-how for the canonical procedure.

- [x] **Step 1: Preconditions**

```bash
cd /home/apiad/Workspace/repos/sixma
git rev-parse --abbrev-ref HEAD       # main
git status --porcelain                # empty
git fetch origin && git status -sb    # not behind
```

- [x] **Step 2: Pre-release checks**

```bash
uv run ruff check . && uv run pytest --cov=sixma
```

Both must pass.

- [x] **Step 3: Bump version**

Either `NEW_VERSION=0.5.0 make release` (makefile path) **or** the explicit sequence:

```bash
sed -i 's/^version = .*/version = "0.5.0"/' pyproject.toml
uv sync
grep '^version' pyproject.toml                                    # 0.5.0
uv run python -c "import sixma; print(sixma.__version__)"         # 0.5.0
```

- [x] **Step 4: Commit + tag + push**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(release): v0.5.0"
git tag -a v0.5.0 -m "Release v0.5.0"
git push origin main
git push origin v0.5.0
```

- [x] **Step 5: Create GitHub Release**

```bash
gh release create v0.5.0 --generate-notes --title v0.5.0
```

- [x] **Step 6: Wait for `release.yaml` workflow + verify PyPI**

```bash
gh run watch
gh release view v0.5.0 --json url --jq '.url'
# Then check https://pypi.org/project/sixma/0.5.0/ within ~5min
```

- [x] **Step 7: Smoke test from PyPI**

```bash
cd /tmp && uv init sixma-smoke && cd sixma-smoke
uv add sixma==0.5.0
uv run python -c "import sixma; print(sixma.__version__)"   # 0.5.0
```

---

## Acceptance criteria mapping (spec §4)

| # | Criterion | Verified by |
|---|---|---|
| 1 | All 23 existing tests pass unchanged | Full suite pass at every commit |
| 2 | Vertical-slice Stack test falsifies | `test_vertical_slice_stack_falsifies` |
| 3 | Trace + seed reproducer in failure message | Same test asserts on "Trace:" / "Seed:" |
| 4 | SIXMA_SEED reproduces the trial | `test_seed_reproduces_falsification` |
| 5 | Pick-only trivial invariant certifies | `test_pick_only_test_certifies_trivial_invariant` |
| 6 | Mixed generator + ctx works | `test_mixed_generator_and_ctx` |
| 7 | Divergent body discards, doesn't hang | `test_divergent_body_discards_via_max_picks_per_trial` |
| 8 | Discard counter increments → CertificationError | `test_max_picks_exhaustion_then_certification_error` |

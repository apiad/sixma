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


# --- Decorator integration tests (acceptance criteria 5, 6, plus structural) ---


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
        def _two_ctx(ctx: PickContext, other: PickContext):
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


# --- Vertical-slice Stack/LIFO + safety-valve acceptance (criteria 2,3,4,7,8) ---


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


def _make_stack_lifo():
    @certify(reliability=0.999, confidence=0.95, max_picks_per_trial=100)
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

    return stack_lifo


def test_vertical_slice_stack_falsifies():
    """Acceptance criteria 2 + 3: vertical-slice falsifies and reports a trace."""
    stack_lifo = _make_stack_lifo()
    with pytest.raises(AssertionError) as excinfo:
        stack_lifo()  # type: ignore
    msg = str(excinfo.value)
    assert "Falsified" in msg
    assert "Trace:" in msg
    assert "Seed:" in msg
    assert "SIXMA_SEED=" in msg


def test_seed_reproduces_falsification(monkeypatch):
    """Acceptance criterion 4: SIXMA_SEED reproduces the exact failing trial."""
    import re

    monkeypatch.delenv("SIXMA_SEED", raising=False)
    stack_lifo = _make_stack_lifo()
    with pytest.raises(AssertionError) as first:
        stack_lifo()  # type: ignore
    msg1 = str(first.value)
    m = re.search(r"SIXMA_SEED=(\d+)", msg1)
    assert m, f"seed not found in: {msg1}"
    seed = m.group(1)

    monkeypatch.setenv("SIXMA_SEED", seed)
    with pytest.raises(AssertionError) as second:
        stack_lifo()  # type: ignore
    msg2 = str(second.value)

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

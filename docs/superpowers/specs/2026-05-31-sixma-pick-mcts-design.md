---
date: 2026-05-31
status: implemented (v0.5.0, 32f59c9) — v1 scope shipped; §5 axes still deferred
author: Alex Piad (with Claude Opus 4.7)
scope: sixma v0.5 — pick-based test bodies; full vision captured for later iterations
---

# Sixma: pick-based test bodies + the path to universal statistical correctness

## 1. The vision

Sixma's tagline today is "stop writing unit tests, start certifying reliability."
That promise currently covers one slice — input-domain invariants of pure
functions, certified by zero-failure binomial — but the framing is bigger than
the slice. The real question sixma is positioned to answer is:

> **At what confidence does this software satisfy this claim about its behavior?**

The user states the claim and the desired confidence; sixma decides how many
trials it takes, runs them, and reports certify / falsify / falsified-with-witness.
This is a question every existing testing tool asks badly:

- Hypothesis owns property-based testing but never tells the user "you ran 100
  trials, here's what that means in confidence terms."
- pytest / unittest test individual cases without any space-level claim.
- Jepsen tests linearizability but isn't a general framework.
- pytest-benchmark + scipy own performance + distributions without a unifying claim.
- Mutation testers (stryker, mutmut) measure test-suite sensitivity but don't
  produce a user-stated confidence.
- Concolic tools (CrossHair, Pynguin) explore paths via SMT but don't quantify
  confidence — they aim at exhaustiveness, which is a different framing.

Sixma's bet is that **reliability `R` + confidence `C` is the right anchor
question**, and that once it's the anchor, the rest of "is this software right"
becomes pluggable along three axes:

- **Spaces** — what's varying across trials: inputs, op-traces, interleavings,
  perturbations, implementations, versions.
- **Claims** — what's being asserted: boolean invariants, equivalence to a
  reference, metamorphic relations, distributional matches, performance bounds.
- **Statistical models** — how trials become confidence: zero-failure binomial,
  sequential probability ratio testing, bootstrap CIs, Kolmogorov–Smirnov,
  Bayesian posteriors.

A **Claim** is the triple `(Space, Property, Model)`. Today's `@certify` is
the specific case `(InputSpace, BooleanInvariant, ZeroFailureBinomial)`. The
universal version makes each axis pluggable.

The unifying observation that motivates the v1 in this doc: **every Space
collapses to a tree.** Internal nodes are decision points (input choices,
op selection, schedule decisions, transform choices), edges are outcomes,
leaves are verdicts. Picks make that tree first-class in the test body. Once
the tree is the substrate, swapping uniform sampling for MCTS, importance
weighting, or coverage-guided exploration is an axis you can move along
without rewriting the user-facing API.

This document captures the full vision but only commits to v1 work. The rest
is enumerated in section 5 as deferred axes — not promised, but tracked so
the v1 design doesn't paint into a corner.

## 2. v1 scope

What ships in v0.5:

- A `ctx` parameter the decorator injects into the test body when the body
  declares one. Tests express stateful or coupled behavior naturally via
  `ctx.pick`, `ctx.range`, `ctx.discard`.
- Uniform random sampling at every pick site. Statistical math operationally
  unchanged from today (zero-failure binomial, `N = ⌈ln(1-C)/ln(R)⌉`).
- A `Sampler` abstraction inside the decorator with `UniformSampler` as the
  only implementation. The seam where v2 (MCTS) plugs in without API churn.
- A new failure-report shape for `ctx`-using tests: ordered trace of
  `(label, value)` records.
- A safety-valve knob `max_picks_per_trial=1000` on the decorator. Trials
  exceeding it are discarded internally, preventing the framework from
  hanging on a divergent test body.

What v1 deliberately does **not** ship (see section 5 for the deferred list):

- MCTS or any non-uniform sampler.
- Reframed statistical claim about reachable paths.
- Importance weighting.
- Weighted picks API (`ctx.weighted(...)`), convenience sugar
  (`ctx.assume`, `ctx.element_of`), edge-case-first ordering for `ctx.range`.
- Shrinking for pick-based traces. (Legacy default-value-generator shrinking
  is preserved unchanged.)
- Parallel trial execution.
- Per-pick-site sub-RNGs for refactor-stable seeds.

The v1 line was chosen for two reasons. First, it's a true vertical slice:
every existing sixma test keeps working unchanged, plus a new shape becomes
expressible. Second, shipping the pick primitive with uniform sampling
generates the real test corpus needed to inform later design choices — which
pick patterns are common, what node-identity scheme works, what shrinking
behavior tests actually want.

## 3. v1 design

### 3.1 Architecture & mental model

A trial is one randomized execution of the test body. The body is a Python
function that may take a `ctx` parameter; calling `ctx.pick(...)` /
`ctx.range(...)` mid-execution requests a value from the framework, which
produces it from the trial's seeded RNG via the `Sampler`. Each trial walks
one root-to-leaf path through the decision tree the test body implicitly
defines. The leaf is a verdict (pass / fail / discard).

`@certify(R, C)` runs trials until one of:

- `N = ⌈ln(1-C)/ln(R)⌉` non-discard trials have passed → **certified**.
- A trial raises `AssertionError` → **falsified** with trace + seed.
- The discard counter exceeds `max_discards` → **CertificationError**.

The mental shift from today: the framework no longer generates the input
before invoking the body. The body is itself the generator — it pulls values
from `ctx` as control flow demands.

Three new internal pieces orchestrated by the existing decorator:

- **`Trial`** — one execution of the body. Owns the trial's pick records (an
  ordered list of `(label, value)` tuples), a pick counter (for the safety
  valve), and a verdict slot. Created per trial; not user-facing.
- **`Sampler`** — interface with one method: `choose(options, label,
  trial) -> chosen`. v1 has `UniformSampler` calling `trial.rng.choice(options)`
  or `trial.rng.randint(low, high)`. v2 will add `MCTSSampler` here.
- **`PickContext`** — the `ctx` object the user touches. Thin facade over
  `Trial` and `Sampler`. Each call records into `Trial` and delegates the
  random decision to `Sampler`.

The existing decorator path (default-value generators bound at decoration
time) is untouched. Parameter inspection at decoration time decides which
path runs: any parameter with no `BaseGenerator` default / no `Annotated`
generator hint / no `BaseGenerator`-subclass type is the `ctx` slot. If no
such parameter exists, the legacy path runs unchanged.

### 3.2 Pick API

Three primitives on `ctx`:

```python
def pick(self, *options: T, label: str | None = None) -> T: ...
def range(self, low: int, high: int, label: str | None = None) -> int: ...
def discard(self) -> NoReturn: ...
```

**`pick(*options, label=None)`** — exactly one of `options` is returned by
uniform random choice. Requires at least one option; zero-option calls raise
`ValueError`. mypy-friendly via overload: when all positional args share a
type `T`, the return type is `T`; if types diverge, return is `Any`.

**`range(low, high, label=None)`** — integer in `[low, high]` inclusive,
sampled uniformly. v1 is pure uniform random. Edge-case-first ordering
(yielding `low`, `high`, `0`, `±1` before random) is deferred — see section
5, axis D — because the edge-first behavior of the existing `_Integer`
generator is what makes its sampling not actually uniform, and we want v1's
contract to be the simplest defensible thing.

**`discard()`** — raises a private `_Discard` sentinel exception that the
decorator's trial loop catches as a precondition failure. Equivalent to
`require(False)`; both share machinery. `require()` remains documented for
tests that don't use `ctx`; `ctx.discard()` is the idiomatic form for tests
that do.

**Labels** are reporting affordances in v1. If `label=None`, the framework
auto-assigns `pick_<ordinal>` (zero-indexed by call order within this trial).
Explicit labels are stored verbatim and surface in failure traces. When v2
MCTS lands, labels become cross-trial pick-site identity; unlabeled picks
will fall back to source `file:line` at that point. Users who want
MCTS-friendly tests should label their picks now, but it's optional in v1.

### 3.3 Decorator integration

`@certify` gains one keyword:

```python
@certify(
    reliability=0.999,
    confidence=0.95,
    max_discards=10000,
    max_picks_per_trial=1000,
)
def test_account(ctx):
    ...
```

`max_picks_per_trial` defaults to `1000`. The `PickContext` counts every call
to `pick` / `range`; the (max+1)-th call raises an internal `TrialTooDeep`
exception that the decorator treats as a discard. Real tests typically take
5–50 picks per trial.

Parameter inspection at decoration time gains one extra step before the
existing two:

1. Identify the `ctx` parameter: at most one parameter with no `BaseGenerator`
   default, no `Annotated[..., BaseGenerator]` hint, and no `BaseGenerator`-subclass
   type. Raise `ValueError` at decoration if more than one is found.
2. (existing) Identify generator-default parameters.
3. (existing) Identify `Annotated[…]`-hinted parameters.
4. (existing) Identify raw-instance-hinted parameters.

If a `ctx` parameter is found, every trial constructs a fresh `PickContext`
(bound to the trial's RNG and `Sampler`) and passes it as the `ctx` argument.
If not, the legacy path runs unchanged.

Tests may **mix** mechanisms — a body that takes both a `ctx` and a
default-value generator works:

```python
@certify
def test_mixed(x: int = g.Integer(0, 100), ctx=...):
    while ctx.pick(True, False):
        x += ctx.range(-10, 10)
    assert -10000 < x < 10000
```

The default-value generator is bound once per trial as today; the `ctx` is
constructed alongside it; both flow into the body.

### 3.4 Statistical contract

Operationally unchanged: zero-failure binomial, `N = ⌈ln(1-C)/ln(R)⌉`
non-discard trials must pass for certification. Trials are i.i.d. (uniform
sampling guarantees this), so the binomial holds.

What shifts is **what R means** for `ctx`-using tests:

- **Legacy claim** (default-value generators): with confidence ≥ `C`, at
  least fraction `R` of the *input space* satisfies the property.
- **Pick-based claim**: with confidence ≥ `C`, at least fraction `R` of the
  *paths the test body explores under uniform `ctx` choices* satisfies the
  property. Equivalently: the probability that a randomly executed trial
  passes is ≥ `R` at confidence `C`.

The two coincide for tests that use only default-value generators (the path
distribution there *is* the input distribution). They diverge for tests
using picks, because the path distribution is induced by the body's own
control flow.

**The semantic responsibility shift, documented plainly.** Under picks, `R`
no longer characterizes an external input space; it characterizes the
test-body-as-random-program. The author owns making that distribution
meaningful. Tests with deep failures behind unlikely pick sequences (e.g.,
a bug that only surfaces after 50 binary picks all going one way) are
exponentially under-sampled by uniform sampling and need either weighted
picks (a v2 feature) or restructured bodies. This is the same trade
Hypothesis users make with `data.draw()`. The README and docstring for
`@certify` will state this explicitly rather than hide it.

**Discard accounting** is unchanged. Discarded trials don't count toward `N`
or the failure tally; they count only against `max_discards`. The
certification claim is conditional: "of trials that don't discard, at least
`R` fraction pass." This matches user intent — discards represent
precondition failures, not invariant violations.

### 3.5 Reproducibility & seeding

Keep the current model unchanged. The decorator constructs one
`random.Random(seed)` per `@certify` invocation (seeded from `SIXMA_SEED`
env var, or `random.getrandbits(32)` if absent). That single RNG is
consumed sequentially across all trials and all picks within trials.

Same seed + same source → same sequence of trials → same path inside each
trial → same falsification (if any). The existing
`SIXMA_SEED=<n> pytest …` reproducer continues to work for both legacy
and pick-based tests.

**Labels don't affect the RNG.** A pick consumes one
`rng.choice` / `rng.randint` call regardless of whether it has an explicit
label. Renaming labels or adding `label=` arguments is RNG-neutral.

**Refactoring the body breaks reproducibility, by design.** Reordering picks,
adding a new pick at the top, or changing a loop bound shifts how the RNG
sequence aligns with control flow. The same seed reproduces a different
trial in the modified body. This matches current behavior. Section 5
axis F notes the v2 path to per-pick-site sub-RNGs for refactor-stable
seeds.

**Parallel trial execution is out of scope for v1.** Sixma runs trials
sequentially today; we keep that. Section 5 axis L tracks the future.

### 3.6 Reporting & failure messages

The failure surface for the new path:

```
❌ Falsified at trial 412!
   Seed: 84920174 (Set SIXMA_SEED=84920174 to reproduce)
   Trace:
     pick_0 = "deposit"
     pick_1 = 500          # ctx.range(1, 1000)
     pick_2 = "withdraw"
     pick_3 = 600          # ctx.range(1, 1000)
     pick_4 = "stop"
   Error: AssertionError: assert account.balance >= 0
```

The trace is the ordered list of `(label, value)` records from the
`PickContext`. Each entry shows the label (user-supplied or auto-assigned
`pick_<N>`) and the chosen value. Long traces (more than 20 picks) are
truncated in the middle of the display with `...`; the full trace is
always available as `ctx.trace` for in-debugger inspection.

The legacy-path failure message is unchanged from today's
`Inputs: {x: -5}` shape. The two formatters share a helper but produce
different surfaces — generators get a single-line `Inputs:` dict; picks
get a multi-line `Trace:` block. The decorator chooses based on whether
the trial used `ctx`.

Logger behavior is unchanged: same `[Sixma] Target: N successes (R=…, C=…)`
info line at the start of a `@certify` run, `[Sixma] Certified ✔️ (N passed)`
on success, the falsification block above on failure.

**No new error types in v1.** `CertificationError` still wraps both
"too many discards" and the new "trial too deep" condition (the latter is
treated as a discard internally). Splitting the error type for clarity is
polish, not v1 scope.

### 3.7 Shrinking

**The legacy default-value-generator path keeps its current one-shot
shrinking** (the `next(iter(gen_inst))` minimal-case re-run that's already
implemented in `core.py:137-150`). It works, it's been shipped, removing it
is a regression. Unchanged in v1.

**The new pick-based path has no shrinking in v1.** A falsified
pick-based trial reports the random trace + seed; users reproduce via
`SIXMA_SEED` and debug from there. Adaptive shrinking is bundled with the
MCTS work as a v2 axis (section 5 axis E) — both are about navigating the
same tree, so deferring them together is natural.

## 4. v1 vertical slice & acceptance criteria

The thinnest end-to-end test of the v1 system: a deterministic finite state
machine with an injected bug, certifying invariants via pick-based op
traces, where the framework finds the bug and reports a useful trace.

```python
# tests/test_pick_vertical_slice.py

from sixma import certify

class Stack:
    def __init__(self):
        self._items = []
    def push(self, x):
        self._items.append(x)
    def pop(self):
        # Injected bug: pops from the wrong end if length == 3.
        if len(self._items) == 3:
            return self._items.pop(0)
        return self._items.pop()
    def size(self):
        return len(self._items)

@certify(reliability=0.99, confidence=0.95, max_picks_per_trial=100)
def test_stack_lifo(ctx):
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
```

Acceptance criteria for v0.5 release:

1. All existing sixma tests pass unchanged.
2. The vertical-slice test above falsifies (the bug at `len == 3` surfaces).
3. The failure trace shows the sequence of picks that produced the failure
   and the seed for reproduction.
4. `SIXMA_SEED=<reported seed>` reproduces the exact failing trial.
5. A test that uses only `ctx` (no default-value generators) certifies a
   trivially-true invariant within the expected `N` trials.
6. A test that mixes default-value generators and `ctx` works correctly.
7. A test that loops forever on picks (`while True: ctx.pick(1, 2)`) does
   NOT hang the framework — `max_picks_per_trial` kicks in and the trial
   discards.
8. `max_picks_per_trial` exhaustion increments the discard counter; when
   `max_discards` is reached, `CertificationError` surfaces.

## 5. Deferred axes (the broader vision)

Each axis below is a real piece of "universal statistical correctness"
that sixma is positioned to absorb. None are v1 commitments; they're
tracked here so the v1 design doesn't paint into a corner.

### Axis A — MCTS sampler

Replace `UniformSampler` with a Monte Carlo Tree Search engine that
allocates trials adaptively: UCB1 (or Thompson sampling) at each pick
site balances exploration (visit under-sampled branches) against
exploitation (revisit branches where failures or discards are concentrated).
Pick-site identity comes from labels (explicit) or source `file:line`
(implicit fallback). The tree is in-memory, per `@certify` invocation;
discarded if the framework restarts.

The interface seam is already carved in v1: `Sampler.choose(options, label,
trial)`. `MCTSSampler` slots in here. Operationally invisible to the user
beyond a new decorator knob (`sampler="mcts"` or similar).

### Axis B — Reframed statistical claim

Today's claim under picks is "fraction of the body's induced path
distribution that passes." A stronger framing is "fraction of *reachable*
paths that pass" (uniform-over-paths, not uniform-over-sampling-distribution).
This is more philosophically defensible — it's what users actually want when
they ask "is any reachable behavior wrong?" — but requires either
importance-weighting MCTS samples or treating the tree as a path-coverage
substrate with a depth bound. Mathematically nontrivial; defer until A
ships and we see real-world MCTS behavior.

### Axis C — Search-then-certify split

When B is too ambitious, the compromise is a two-phase trial loop: phase 1
MCTS hunts for counterexamples within a search budget; if found, falsify.
Phase 2 reverts to uniform sampling for `N` trials and applies the standard
zero-failure binomial. This keeps the math clean while letting MCTS into
the system. May ship before or alongside A depending on how the tree-shape
research goes.

### Axis D — Weighted picks + convenience sugar

API additions:

- `ctx.weighted([(opt, weight), …], label=None)` — weighted pick; useful
  for biasing termination (`weighted([("stop", 1), ("continue", 4)])`).
- `ctx.assume(cond)` — sugar for `if not cond: ctx.discard()`.
- `ctx.element_of(collection, label=None)` — sugar for `ctx.pick(*collection)`.
- `ctx.range_float(low, high, label=None, allow_nan=False, allow_inf=False)`.
- `ctx.boolean(label=None)` — sugar for `ctx.pick(True, False)`.
- Edge-case-first ordering for `ctx.range` (matching `_Integer`'s current
  behavior) — possibly opt-in via `ctx.range(low, high, edges_first=True)`.

Additive — no v1 design churn. Likely a v1.1 patch release.

### Axis E — Adaptive shrinking

When a pick-based trial falsifies, automatically search for a smaller
trace that also falsifies: shrink each pick toward its first option,
remove trailing picks, binary-search the depth. Hypothesis-style.
Coupled to A because both navigate the same tree.

### Axis F — Per-pick-site sub-RNGs

Today, one RNG per `@certify` invocation drives the whole sequence.
Refactoring the body realigns the RNG with control flow and breaks
seed reproducibility. v2 may derive a sub-RNG per labeled pick site from
`hash(master_seed, label)`. Refactor-stable seeds for labeled picks;
unlabeled picks fall back to the v1 sequential behavior.

### Axis G — Concurrent / scheduling tests

Schedule decisions as picks: `ctx.pick("thread_a", "thread_b")` chooses
which thread runs next, then the framework executes one step of that
thread. Linearizability checking falls out — at each step, verify the
observable sequence is consistent with some sequential history.
Significant complexity (thread harness, atomicity boundaries, history
recording); not a near-term axis.

### Axis H — Equivalence and metamorphic claims

First-class decorators that take two callables (or one callable and a
transform) and certify equivalence over a pick space:

```python
@certify_equivalent(impl=fast_sort, reference=sorted)
def test_sort_eq(ctx):
    items = [ctx.range(0, 100) for _ in range(ctx.range(0, 20))]
    return (items,)

@certify_metamorphic(f=parser.parse, relation=lambda x, y: x.tree == y.tree)
def test_parser_idempotent(ctx):
    src = ctx.pick("a + b", "(a)", "x = 1; y = 2")
    return (src, ctx.pick("verbatim", "double_whitespace", "trailing_nl"))
```

The decorator signature is the API; the statistical model is still
zero-failure binomial. Useful for sciencey / ML / refactor work where
there's no oracle but two implementations exist.

### Axis I — Distributional claims

`@certify_distribution(distribution=scipy.stats.norm, alpha=0.01)` — the
test body produces one sample per trial; the framework collects them and
applies Kolmogorov–Smirnov / chi-square / Anderson–Darling to certify the
samples match the claimed distribution at significance `alpha`. For RNG
testers, samplers, Monte Carlo simulators. Different statistical model
than zero-failure binomial — but the trial-loop machinery is shared.

### Axis J — Performance claims

`@certify_performance(p99 < timedelta(milliseconds=50))` — the test body
is timed; the framework collects timings and certifies the bound at user
confidence. Needs careful repeat/median/outlier-trim machinery to handle
system noise (JIT, GC, scheduler). Per user instruction, this axis is
deferred indefinitely — pinned here for completeness, not prioritized.

### Axis K — Sequential testing (SPRT)

Replace zero-failure binomial with Wald's Sequential Probability Ratio
Test. Trial count adapts to the data — decisively-passing tests stop
early; ambiguous tests run longer. Typically 2–3× tighter trial budgets
for the same `R`, `C`. Lives alongside `ZeroFailureBinomial` as an
optional `Model` plugin.

### Axis L — Parallel trial execution

Trials are embarrassingly parallel modulo the shared RNG. With per-trial
RNGs derived deterministically from a master seed, trials can run in
processes / threads. Speedup is linear in core count. Needed when
`N > 10000` for high-confidence tests.

### Axis M — Coverage-guided generation

Beyond MCTS, integrate code-coverage feedback (`coverage.py`,
`sys.settrace`) to direct trial allocation toward unexplored branches of
the system-under-test, not just the test body. Bridges sixma into
coverage-guided fuzzing territory (AFL-style). Requires statistical
machinery similar to B (importance weighting).

### Cross-cutting: claim composition

A long-arc axis: claims about modules compose into claims about systems.
If `f` is certified at `R_f, C_f` and `g` is certified at `R_g, C_g`,
what can we say about `g ∘ f`? Bayesian-flavored compositional reasoning.
Open research direction; not a near-term axis but the destination this
whole framework is pointed at.

## 6. Open questions

These are unresolved design points that the v1 implementation will surface
answers to. Capturing them so the implementation plan has explicit decision
points.

1. **`ctx.trace` exposure** — should `ctx.trace` be a public attribute users
   can read mid-test (to assert on the path taken so far), or strictly an
   internal-for-reporting field? Default: internal in v1; promote to public
   if real tests want it.

2. **Long-trace truncation policy** — exact format for `>20 picks` traces.
   Tail-only? Head + tail? User-configurable? Default in v1: show first 10 +
   `... (X picks elided)` + last 5.

3. **`max_picks_per_trial=1000` as the right default** — too generous, too
   tight? Will revisit after the vertical-slice test corpus matures.

4. **mypy ergonomics of `ctx.pick(*options)`** — the `T`-overload claim above
   may not actually work cleanly for heterogeneous types. May need to fall
   back to `Any` return more often than is ideal. Verify during
   implementation.

5. **`PickContext` thread-safety** — irrelevant in v1 (sequential trials)
   but trial-running-in-thread is a v2 axis. Note as future concern.

## 7. Out of scope (clarification list)

These items are sometimes-asked-about and are explicitly out of v1:

- No statistical claim *about* `max_picks_per_trial` — it's a safety valve,
  not a coverage knob.
- No `pytest`-plugin integration. `@certify` keeps working as a plain
  decorator; pytest-level reporting (e.g., custom markers, JUnit XML
  enhancement) is a separate axis.
- No CHANGELOG.md introduction. (Separate axis; tracked in
  `know-how/releasing.md` for whenever Alex wants to start one.)
- No mypy gate in CI. (Separate axis.)

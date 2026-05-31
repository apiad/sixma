# sixma — agent orientation

Statistical testing framework for Python. Replaces hand-written unit cases with **generative spaces** + **zero-failure certification**: you state an invariant + target reliability `R` and confidence `C`, and `@certify` runs `N = ⌈ln(1-C)/ln(R)⌉` randomized trials, failing fast on the first counter-example with seed + shrunk minimal input.

Small repo — one module of decorator + one module of generators. Public surface lives in `sixma/` and is re-exported from `sixma/__init__.py`.

## Running

    uv sync                          # install dev deps (pytest, ruff, coverage)
    uv run pytest -q                 # full test suite (23 tests, ~0.3s)
    uv run pytest --cov=sixma        # with coverage (currently 80%)
    uv run ruff check .              # lint gate (CI runs this before tests)
    uv run ruff format .             # autoformat

CI: `.github/workflows/tests.yaml` (push/PR to main) and `release.yaml` (on
GitHub Release creation → tests + PyPI publish via `uv publish`). Both pin
Python 3.13 and install uv inline.

## Layout

- `src/sixma/core.py` — the `@certify(reliability, confidence, max_discards)`
  decorator + `require()` precondition + `PreconditionError` / `CertificationError`.
  The decorator does parameter discovery (default-value `BaseGenerator`
  instances, then `Annotated[T, gen]` fallback, then raw-instance fallback),
  builds per-trial RNG-bound streams, runs the trial loop until `N` successes
  or `max_discards`, and on `AssertionError` re-runs with each generator's
  first-yielded value to surface a shrunk minimal example.
- `src/sixma/generators.py` — generator hierarchy. `BaseGenerator` is the root;
  internal classes are `_Integer`, `_Float`, `_Bool`, `_String`, `_Date`,
  `_DateTime`, `_List`, `_Dict`, `_Object`, `_Case`. Public factory functions
  at the bottom return `cast(T, _Internal(...))` so mypy sees the target type
  (`Integer(0, 10) -> int`) while the runtime value is a generator instance
  the decorator detects via `isinstance(_, BaseGenerator)`. Each generator
  yields edge cases first (`0`, `1`, `-1`, bounds, leap days, empty
  collections, etc.) before switching to random sampling via the shared
  `smart_sample` helper (10% chance to re-pick an edge case mid-stream).
- `src/sixma/__init__.py` — public re-exports: `certify`, `require`, and
  `__version__`.
- `tests/` — 6 files. `test_core.py` covers the certification loop, shrinking,
  seeding, precondition discards. `test_combinators.py` / `test_case.py` /
  `test_datetime.py` cover individual generators. `test_api_shortcuts.py` and
  `test_new_syntax.py` cover the three parameter-declaration syntaxes
  (default-value, Annotated, raw-instance).
- `makefile` — `test-unit`, `test-all`, `format`, `release`. See the release
  know-how before invoking `release` — the existing target has gaps.
- `know-how/` — operational procedures. See index below.

## Conventions

- **The `_Internal` / `Public` factory split is the trick that keeps mypy
  happy.** When adding a new generator, mirror the pattern: define
  `_Foo(BaseGenerator)` (with `__init__`, `bind(rng)`, `__iter__` yielding
  edges-then-samples, and `sample()`), then export a public `Foo(...) -> T`
  factory that does `return cast(T, _Foo(...))`. The `cast` is what lets users
  write `x: int = g.Foo(...)` without mypy complaining.
- **`bind(rng)` returns a fresh generator instance bound to the trial's RNG.**
  Generators are otherwise RNG-less; the decorator binds them per-trial so the
  same `SIXMA_SEED` deterministically reproduces a failure.
- **Edge cases yield first, sampling after.** This is intentional — most bugs
  hide at boundaries, so `__iter__` yields the `_edges` list before entering
  the `while True: yield self.sample()` random tail. Preserve this ordering.
- **Tests are plain pytest.** A `@certify` function is callable as `f()` with
  no args (the decorator strips sixma params from the signature). `# type: ignore`
  is acceptable at the call site since mypy doesn't understand the rewrite.
- **Python 3.12+** per `pyproject.toml` (CI runs 3.13). `uv` is the package
  manager; do not use `pip` directly.
- **Commit straight to main** (per workspace convention for solo repos —
  feature-branch+PR only when explicitly requested). Conventional commits.

## Know-how

- [releasing.md](know-how/releasing.md) — cut a new version + publish to PyPI.
  Reach for it when bumping `pyproject.toml` and tagging. The makefile
  `release` target has gaps (no dirty-tree check, sed on `__init__.py` silently
  no-ops since `__version__` is wired to `importlib.metadata`); follow the
  know-how, not the makefile.

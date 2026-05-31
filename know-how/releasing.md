# Releasing a new sixma version

**When to reach for this:** you've merged work to `main` and want to cut a
new PyPI release. Replaces the `release` target in `makefile` (which has
gaps — see [Gaps in the makefile target](#gaps-in-the-makefile-target)
below).

## TL;DR

    # from /home/apiad/Workspace/repos/sixma, on main, clean tree
    NEW_VERSION=x.y.z make release   # the makefile path (safe after the fixes below)

    # or step-by-step, agent-friendly (the canonical sequence)
    git status --porcelain | grep -q . && echo "DIRTY — abort" && exit 1
    uv run ruff check . && uv run pytest --cov=sixma
    sed -i "s/^version = .*/version = \"x.y.z\"/" pyproject.toml
    uv sync
    git add pyproject.toml uv.lock
    git commit -m "chore(release): vx.y.z"
    git tag -a "vx.y.z" -m "Release vx.y.z"
    git push origin main && git push origin "vx.y.z"
    gh release create "vx.y.z" --generate-notes --title "vx.y.z"

The GitHub Release fires `.github/workflows/release.yaml`, which runs tests
then `uv publish` with `PYPI_TOKEN` — that's how sixma reaches PyPI.

## Procedure (atomic, cheapest-first)

Ordered so any failure before step 6 leaves no observable state change.
Steps 6–9 are the irreversible ones — they go last so the cheap checks
have already passed.

### 1. Preconditions

    cd /home/apiad/Workspace/repos/sixma
    git rev-parse --abbrev-ref HEAD          # must be 'main'
    git status --porcelain                   # must be empty
    git fetch origin && git status -sb       # must NOT be 'behind'

If any check fails, stop. Don't release on a non-default branch, with a
dirty tree, or behind origin.

### 2. Decide the version bump

Look at commits since the last tag:

    LAST_TAG=$(git describe --tags --abbrev=0)
    git log "$LAST_TAG..HEAD" --pretty=format:'%s'

Classify per conventional commits:

- Any `feat!:` / `fix!:` / `BREAKING CHANGE:` → **major**
- Any `feat:` → **minor**
- Otherwise → **patch**

State the proposed version explicitly before proceeding.

### 3. Pre-release checks

    uv run ruff check .
    uv run pytest --cov=sixma

Both must pass. Coverage is informational; ruff and tests are gates.

### 4. Bump the version

`pyproject.toml` is the **single source of truth** for the version. Since
`fix(version): resolve __version__ via importlib.metadata`, the runtime
`sixma.__version__` reads from package metadata at import time, so there
is nothing to bump in `src/sixma/__init__.py`.

    sed -i "s/^version = .*/version = \"x.y.z\"/" pyproject.toml
    uv sync                                  # refreshes uv.lock with the new pin

Verify:

    grep '^version' pyproject.toml           # version = "x.y.z"
    uv run python -c "import sixma; print(sixma.__version__)"   # x.y.z

### 5. CHANGELOG (when one exists)

sixma does not yet maintain a `CHANGELOG.md`. If/when one is added,
follow Keep-a-Changelog: rename the `[Unreleased]` section to
`[vX.Y.Z] - YYYY-MM-DD` and insert a fresh `[Unreleased]` above.

### 6. Commit, tag, push

    git add pyproject.toml uv.lock
    git commit -m "chore(release): vx.y.z"
    git tag -a "vx.y.z" -m "Release vx.y.z"
    git push origin main
    git push origin "vx.y.z"

**Never `--force` push** and never overwrite an existing tag. If the push
rejects with non-fast-forward, stop and investigate — the branch has
diverged from origin since step 1.

### 7. Create the GitHub Release

    gh release create "vx.y.z" --generate-notes --title "vx.y.z"

This fires `.github/workflows/release.yaml`. The workflow:

1. Runs `ruff check` + `pytest --cov` against Python 3.13.
2. Runs `uv build` to produce wheel + sdist.
3. Runs `uv publish --token $PYPI_TOKEN` to push to PyPI.

If the workflow fails, the tag and GitHub Release still exist — that's
expected; fix forward (next patch release) rather than retracting.
Tags are immutable.

### 8. Verify on PyPI

Within a minute or two:

    open https://pypi.org/project/sixma/x.y.z/       # or `gh release view vx.y.z`

`uv add sixma==x.y.z` should resolve in a fresh project.

## Gaps in the makefile target

The current `release` target (as of v0.4.0) has these issues. The
procedure above addresses each one; the makefile should be updated to
match.

| Gap | Fix |
| --- | --- |
| No `git status --porcelain` check at step 1 — releases on a dirty tree are silently allowed. | Add `@git diff --quiet --exit-code \|\| { echo "Dirty tree"; exit 1; }` at the top. |
| `sed -i.bak "s/__version__ = ...` on `src/sixma/__init__.py` no-ops silently. The `__version__` line no longer exists (it's now read from `importlib.metadata`). | Remove the sed and the matching `rm`/`git add` lines. |
| `git add src/sixma/__init__.py` adds nothing, but masks intent. | Remove. |
| Commit message `Bump version to $(NEW_VERSION)` doesn't follow conventional commits. | Use `chore(release): v$(NEW_VERSION)`. |
| `git push` / `git push --tags` are remote-less and tag-list-less — they work today via tracked upstream but are brittle. | `git push origin main` and `git push origin "v$(NEW_VERSION)"` explicit. |
| No branch check — `make release` from a feature branch produces a tag on the wrong branch. | Add `@[ "$$(git rev-parse --abbrev-ref HEAD)" = "main" ] \|\| { echo "Not on main"; exit 1; }`. |

These fixes are applied in the `makefile` itself; this know-how doc is
the source of truth for the procedure, and the makefile is a thin
helper that follows it.

## Reproducing a release locally (no PyPI push)

To dry-run the build without publishing:

    uv build                                 # wheel + sdist into dist/
    uv run python -c "import sixma; print(sixma.__version__)"

Inspect `dist/sixma-x.y.z*` to confirm the version is what you expect
before tagging.

default: test-unit

.PHONY: docs
docs:
	cd docs && quarto publish gh-pages

.PHONY: bugfix
bugfix:
	git commit -a -m "Bugfix"

.PHONY: format
format:
	ruff format . && git commit -am "Apply code formatting"

.PHONY: format-check
format-check:
	ruff check .

.PHONY: type-check
type-check:
	mypy

.PHONY: test-unit test-all

test-unit: format-check
	pytest tests --cov=sixma

test-all: format-check
	pytest --cov=sixma

.PHONY: docker-build
docker-build:
	docker build -t sixma:latest -f dockerfile .

.PHONY: clean
clean:
	rm -rf dist
	rm -rf sixma_dsl.egg-info
	rm -rf *.db*
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

.PHONY: issues
issues:
	gh md-issues push
	sleep 5
	gh md-issues pull
	git add issues && git commit -m "Sync issues"

# Get the current version from pyproject.toml
CURRENT_VERSION := $(shell grep 'version = ' pyproject.toml | cut -d '"' -f 2)

.PHONY: release
# Procedure documented in know-how/releasing.md (source of truth).
# This target is a thin helper that follows it.
release: format-check
	@if [ -z "$(NEW_VERSION)" ]; then \
		echo "ERROR: NEW_VERSION environment variable is not set."; \
		echo "Usage: NEW_VERSION=x.y.z make release"; \
		exit 1; \
	fi
	@[ "$$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "ERROR: not on main"; exit 1; }
	@git diff --quiet --exit-code || { echo "ERROR: dirty tree, commit or stash first"; exit 1; }
	@git diff --quiet --cached --exit-code || { echo "ERROR: staged changes present"; exit 1; }
	@echo "Current version: ${CURRENT_VERSION} → $(NEW_VERSION)"
	@make test-all

	@echo "Bumping pyproject.toml..."
	@sed -i.bak "s/version = \"$(CURRENT_VERSION)\"/version = \"$(NEW_VERSION)\"/" pyproject.toml
	@rm pyproject.toml.bak

	@uv sync --all-extras

	@echo "Committing version bump..."
	@git add pyproject.toml uv.lock
	@git commit -m "chore(release): v$(NEW_VERSION)"

	@echo "Tagging new version..."
	@git tag -a "v$(NEW_VERSION)" -m "Release v$(NEW_VERSION)"

	@echo "Pushing commit and tag..."
	@git push origin main
	@git push origin "v$(NEW_VERSION)"

	@echo "Creating GitHub release..."
	@gh release create "v$(NEW_VERSION)" --title "v$(NEW_VERSION)" --generate-notes

	@echo "✅ v$(NEW_VERSION) released. PyPI publish runs via release.yaml workflow."

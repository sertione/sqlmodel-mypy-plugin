.DEFAULT_GOAL := help

.PHONY: help install lint fmt fmt-check typecheck test check update-mypy build clean precommit

##@ Project

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} \
		/^[a-zA-Z0-9_.-]+:.*##/ { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

install: ## Install (locked) dev dependencies with uv.
	uv sync --locked --group dev

lint: ## Run ruff lints.
	uv run ruff check .

fmt: ## Auto-format code with ruff.
	uv run ruff format .

fmt-check: ## Check formatting (no changes).
	uv run ruff format --check .

typecheck: ## Run mypy on plugin code.
	uv run mypy

test: ## Run tests (with >=90% coverage gate).
	uv run pytest -n auto --dist=loadscope --cov=sqlmodel_mypy --cov-report=term-missing --cov-fail-under=90

check: fmt-check lint typecheck test ## Run all quality gates (what CI runs).

update-mypy: ## Regenerate golden mypy outputs.
	uv run pytest --update-mypy

build: ## Build sdist/wheel.
	uv build

clean: ## Remove common local caches/build artifacts.
	rm -rf \
		.mypy_cache .pytest_cache .ruff_cache \
		coverage htmlcov .coverage .coverage.* \
		dist build

precommit: ## Run pre-commit hooks on all files.
	uv run pre-commit run --all-files

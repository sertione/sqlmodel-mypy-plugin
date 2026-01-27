# AGENTS.md

# AGENTS.md (repo workflow & invariants)

This repo builds a **mypy plugin** for **SQLModel**.

## Non-negotiable quality gates (every session / every PR)

Before you consider work "done", you **must** have a green:

```bash
make check
```

This includes **tests** and a **coverage gate**: **90%+** for the `sqlmodel_mypy` package.

If your change affects mypy diagnostics or generated signatures, you **must** update and commit golden outputs:

```bash
make update-mypy
```

## Toolchain / stack

- **Language**: Python (plugin code + tests)
- **Runtime**: Python **>= 3.10** (CI tests multiple versions)
- **Package manager / runner**: `uv` (lockfile-driven, reproducible)
- **Lint/format**: `ruff`
- **Typecheck**: `mypy` (plugin uses mypy internal APIs)
- **Tests**: `pytest`
- **Build backend**: `hatchling`

`uv.lock` is the source of truth for dependency resolution in CI and development.

## Quick commands (canonical)

- **Help**: `make help`
- **Install (locked dev env)**: `make install`
- **Format**: `make fmt`
- **Lint**: `make lint`
- **Typecheck**: `make typecheck`
- **Tests**: `make test`
- **All gates (what CI runs)**: `make check`
- **Update golden mypy outputs**: `make update-mypy`
- **Build artifacts**: `make build`

## What the plugin does (scope)

- Collects SQLModel fields and determines required/optional kwargs based on `sqlmodel.Field(...)`.
- **Ignores** `sqlmodel.Relationship(...)` when generating constructor kwargs.
- Synthesizes `__init__` and `model_construct` signatures for SQLModel subclasses during mypy semantic analysis.

## Repo layout (where things live)

- `src/sqlmodel_mypy/plugin.py`: mypy plugin entrypoint, config parsing, cache invalidation.
- `src/sqlmodel_mypy/transform.py`: SQLModel class transformation (collect fields, generate methods).
- `tests/mypy/`: integration tests that run mypy on small modules and commit merged outputs
  - `tests/mypy/configs/`: mypy configs (`*.ini`, `*.toml`)
  - `tests/mypy/modules/`: tiny python modules checked by mypy
  - `tests/mypy/outputs/`: golden outputs produced by merging mypy output into sources

## Invariants / gotchas (do not break these)

- **Plugin ordering matters**: if a user enables both `sqlmodel_mypy.plugin` and `pydantic.mypy`, this plugin must come **first**, otherwise Pydantic's plugin may "claim" SQLModel classes (see `README.md`).
- **Incremental mode safety**:
  - Avoid global state.
  - Persist per-class data in `TypeInfo.metadata` under a **JSON-serializable** key.
  - Current key: `sqlmodel-mypy-metadata` (see `src/sqlmodel_mypy/transform.py`).
- **Idempotency**: class hooks may run multiple times. Code must tolerate re-entry and partial analysis.
  - When types/defs aren't ready, call `api.defer()` and exit early.
- **Cache invalidation is mandatory**:
  - Mypy caches plugin-produced metadata. If you change plugin behavior, bump `__version__` in `src/sqlmodel_mypy/plugin.py`.
  - This value is included in `report_config_data()` and forces mypy to invalidate its cache.

## Golden test policy (how we lock down behavior)

- If you change expected mypy messages/signatures, regenerate outputs with `make update-mypy` and commit the changes under `tests/mypy/outputs/**`.
- Keep test modules minimal and focused: add a new file in `tests/mypy/modules/` and wire it into `tests/mypy/test_mypy.py`.

## Release process (PyPI, trusted publishing)

- Version is defined in `pyproject.toml`.
- To release:
  - Bump the version.
  - Tag `vX.Y.Z` and push the tag.
  - GitHub Actions publishes to PyPI via **trusted publishing** (requires configuring the PyPI project + GitHub environment `pypi`).

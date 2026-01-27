# AGENTS.md

This repo builds a **mypy plugin** for **SQLModel**.

## Quick commands

- **Install (with dev deps)**: `uv sync --group dev`
- **Lint**: `uv run ruff check .`
- **Format**: `uv run ruff format .`
- **Typecheck (plugin code)**: `uv run mypy`
- **Tests**: `uv run pytest`
- **Regenerate golden mypy outputs**: `uv run pytest --update-mypy`

## Repo layout

- `src/sqlmodel_mypy/plugin.py`: mypy plugin entrypoint + hooks
- `src/sqlmodel_mypy/transform.py`: SQLModel class transformer (collect fields, generate `__init__`)
- `tests/mypy/`: integration tests that run mypy against small example modules
  - `tests/mypy/configs/`: mypy config files
  - `tests/mypy/modules/`: python modules checked by mypy
  - `tests/mypy/outputs/`: golden files produced by merging mypy output into sources

## Constraints / gotchas

- **Plugin ordering matters**: if a user enables both `sqlmodel_mypy.plugin` and `pydantic.mypy`, this plugin must come **first**, otherwise Pydantic’s plugin will “claim” SQLModel classes.
- **Incremental mode**: avoid global state. Store per-class data in `TypeInfo.metadata` under a JSON-serializable key.
- **Idempotency**: class hooks may run multiple times. Code must tolerate re-entry and partial analysis; use `api.defer()` when required.
- **Golden tests**: if you change expected mypy messages, run `pytest --update-mypy` and commit the updated `tests/mypy/outputs/**`.

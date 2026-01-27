# Contributing

## Dev setup (uv)

```bash
uv sync --locked --group dev
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

## Mypy plugin integration tests

We test plugin behavior by running mypy against small modules and committing the merged output.

- **Modules**: `tests/mypy/modules/*.py`
- **Configs**: `tests/mypy/configs/*.{ini,toml}`
- **Harness**: `tests/mypy/test_mypy.py`
- **Golden outputs**: `tests/mypy/outputs/**`

### Add a new mypy test case

1. Add a new module under `tests/mypy/modules/` (keep it minimal).
2. Add a `(config, module)` pair to the `cases` list in `tests/mypy/test_mypy.py`.
3. Regenerate outputs:

```bash
uv run pytest --update-mypy
```

4. Ensure the new output contains the expected `# MYPY:` lines (or no output file is created if there are no mypy errors).

## Cache invalidation (important)

Mypy caches plugin-produced metadata. When you change plugin behavior, bump:

- `src/sqlmodel_mypy/plugin.py`: `__version__`

This value is included in `report_config_data()` and forces mypy to invalidate its cache.

## Compatibility policy

- This project uses mypy internal APIs and must be pinned/tested against specific mypy versions.
- `uv.lock` is the source of truth for CI/dev reproducibility.

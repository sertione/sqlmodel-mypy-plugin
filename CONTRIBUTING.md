# Contributing

## Dev setup

```bash
make install
```

## Checks

```bash
make check
```

`make check` includes a **coverage gate**: package coverage must stay **90%+**.

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
make update-mypy
```

4. Ensure the new output contains the expected `# MYPY:` lines (or no output file is created if there are no mypy errors).

## Cache invalidation (important)

Mypy caches plugin-produced metadata. When you change plugin behavior, bump:

- `src/sqlmodel_mypy/plugin.py`: `__version__`

This value is included in `report_config_data()` and forces mypy to invalidate its cache.

## Compatibility policy

- This project uses mypy internal APIs and must be pinned/tested against specific mypy versions.
- `uv.lock` is the source of truth for CI/dev reproducibility.

## Release (PyPI)

This project uses **GitHub Actions** + **PyPI trusted publishing**.

1. Bump the version in `pyproject.toml`.
2. Ensure `make check` is green on `main`.
3. Tag and push:

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
git push --tags
```

4. The `publish` workflow builds and publishes to PyPI.

Maintainers must configure:

- GitHub environment `pypi`
- A trusted publisher on PyPI pointing at this repository/workflow

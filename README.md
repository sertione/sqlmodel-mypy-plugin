# sqlmodel-mypy

Mypy plugin that improves type checking for [`sqlmodel`](https://github.com/fastapi/sqlmodel) models.

## Status

Early/experimental. The first milestone is **correct required/optional kwargs** for fields declared via
`sqlmodel.Field(...)` and **ignoring** `sqlmodel.Relationship(...)` in generated constructor signatures.

## Install

Install for development with `uv`:

```bash
uv sync --group dev
```

## Enable in mypy

`mypy.ini`:

```ini
[mypy]
plugins = sqlmodel_mypy.plugin
```

If you also use `pydantic.mypy`, list this plugin **first** so it can take over SQLModel classes:

```ini
[mypy]
plugins = sqlmodel_mypy.plugin, pydantic.mypy
```

## Plugin configuration

`mypy.ini`:

```ini
[sqlmodel-mypy]
init_typed = false
init_forbid_extra = false
warn_untyped_fields = true
```

`pyproject.toml`:

```toml
[tool.sqlmodel-mypy]
init_typed = false
init_forbid_extra = false
warn_untyped_fields = true
```

## Error codes

- `sqlmodel-field`: field-related plugin errors (e.g. untyped `x = Field(...)`).

## Development

- Lint/format: `uv run ruff check .` and `uv run ruff format .`
- Typecheck plugin code: `uv run mypy`
- Run tests: `uv run pytest`
- Update mypy “golden” outputs: `uv run pytest --update-mypy`

See [`AGENTS.md`](AGENTS.md) for repo conventions and workflow notes.

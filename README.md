# SQLModel Mypy plugin

Mypy plugin that improves type checking for [`SQLModel`](https://github.com/fastapi/sqlmodel) models.

## Status

Early/experimental. The first milestone is **correct required/optional kwargs** for fields declared via
`sqlmodel.Field(...)` and **ignoring** `sqlmodel.Relationship(...)` in generated constructor signatures.

## Install (dev)

Prereqs: **Python \u2265 3.10** and [`uv`](https://docs.astral.sh/uv/).

```bash
make install
```

Run the full quality gate suite (same as CI):

```bash
make check
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

- List available commands: `make help`
- Lint/format: `make lint` / `make fmt`
- Typecheck plugin code: `make typecheck`
- Run tests: `make test`
- Run all quality gates: `make check`
- Update mypy \u201cgolden\u201d outputs: `make update-mypy`

See [`AGENTS.md`](AGENTS.md) for repo conventions and workflow notes.

## Security

See [`SECURITY.md`](SECURITY.md).

## Code of conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

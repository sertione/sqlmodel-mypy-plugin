# SQLModel Mypy plugin

Mypy plugin that improves type checking for [`SQLModel`](https://github.com/fastapi/sqlmodel) models.

## Status

Early/experimental. Current scope:

- Generate correct `__init__` / `model_construct` signatures for SQLModel models (treat `sqlmodel.Field(...)`
  required/optional correctly; ignore `sqlmodel.Relationship(...)` in constructors).
- Improve SQLAlchemy expression typing for **class** attribute access (e.g. `User.id`, `User.name`).

## Install (dev)

Prereqs: **Python 3.10+** and [`uv`](https://docs.astral.sh/uv/).

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

If you also use `pydantic.mypy`, you can list plugins in either order:

```ini
[mypy]
plugins = sqlmodel_mypy.plugin, pydantic.mypy
# or:
# plugins = pydantic.mypy, sqlmodel_mypy.plugin
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

## SQL expression typing

This plugin adjusts **class attribute** types on SQLModel models to behave like SQLAlchemy expressions, so you can
write queries without `col()`:

```py
from sqlmodel import Field, SQLModel, select

class User(SQLModel):
    id: int = Field(primary_key=True)
    name: str = Field()

stmt = select(User).where(User.name.like("%x%"))
```

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

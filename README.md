# SQLModel Mypy plugin

Mypy plugin that improves type checking for [`SQLModel`](https://github.com/fastapi/sqlmodel) models.

## Status

Early/experimental. Implemented scope (supported today):

Planned work is tracked in [`ROADMAP.md`](ROADMAP.md).

- Generate correct `__init__` / `model_construct` signatures for SQLModel models:
  - treat `sqlmodel.Field(...)` required/optional correctly
  - accept `sqlmodel.Relationship(...)` kwargs for `table=True` models
  - accept common `Field(...)` alias kwargs (`alias` / `validation_alias`) when statically known
- Improve SQLAlchemy expression typing for **class** attribute access on `table=True` models (e.g. `User.id`,
  `User.name`, `Team.heroes`).
- Relationship comparator typing on relationship attributes (e.g. `Team.heroes.any(...)`, `Hero.team.has(...)`,
  `Team.heroes.contains(...)`).
- Outer-join `None` propagation in `select(A, B).join(B, isouter=True)` result tuples.
- Broaden `Session.exec()` / `AsyncSession.exec()` typing to accept SQLAlchemy `Executable` statements (e.g.
  `text(...)`), not just `select(...)`.
- Compatible with `pydantic.mypy` in either plugin order.

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

This plugin adjusts **class attribute** types on `table=True` SQLModel models to behave like SQLAlchemy
expressions, so you can write queries without `col()`:

```py
from sqlmodel import Field, SQLModel, select

class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field()

stmt = select(User).where(User.name.like("%x%"))
```

### Relationship comparator typing

Relationship attributes declared via `Relationship(...)` are typed as SQLAlchemy expressions at class level,
including common relationship comparators used in query filters:

```py
stmt = select(Team).where(Team.heroes.any(Hero.name == "x"))
stmt = select(Hero).where(Hero.team.has(Team.name == "t"))
stmt = select(Team).where(Team.heroes.contains(hero_obj))
```

## Typing strategy (defaults)

- **Plugin hooks**: SQLModel-specific behavior (constructor signatures, `table=True` expression typing, etc.).
- **SQLAlchemy typing**: relied upon for core SQL/ORM typing wherever possible.
- **Stub overlays**: only for upstream gaps that can’t be addressed cleanly via hooks (prefer upstream fixes first).

## Field aliases in constructor kwargs

If you use `Field(alias=...)` (or `validation_alias=...` / `schema_extra={"validation_alias": ...}`), the plugin
adds the alias name as an accepted keyword argument in generated `__init__` / `model_construct` signatures.
This avoids false-positive “unexpected keyword argument” errors when `init_forbid_extra=true`.

Limitations:

- Only **string-literal** aliases are recognized (e.g. `Field(alias=\"full_name\")`).
- Aliases that are not valid Python identifiers (or that collide with other parameter names) are ignored.
- Mypy can’t express “either field name or alias is required”, so aliased required fields may not be reported as
  missing at type-check time.

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

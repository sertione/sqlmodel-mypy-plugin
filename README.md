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
- Common SQLAlchemy helpers that operate on expressions / selects (e.g. `.label(...)`, `selectinload(...)`,
  `execution_options(...)`) work without spurious mypy errors.
- Expose SQLAlchemy table metadata on `table=True` models (e.g. `User.__table__`).
- Relationship comparator typing on relationship attributes (e.g. `Team.heroes.any(...)`, `Hero.team.has(...)`,
  `Team.heroes.contains(...)`).
- Outer-join `None` propagation in `select(A, B).join(B, isouter=True)` result tuples.
- Broaden `Session.exec()` / `AsyncSession.exec()` typing to accept SQLAlchemy `Executable` statements (e.g.
  `text(...)`), not just `select(...)`.
- Accept SQLAlchemy TypeEngine instances in `Field(sa_type=...)` (e.g. `DateTime(timezone=True)`, `String(50)`)
  without `# type: ignore` in strict mode.
- Accept `model_config = ConfigDict(...)` overrides on SQLModel subclasses in strict mode.
- Compatible with `pydantic.mypy` **when `sqlmodel_mypy.plugin` is listed first**.

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

If you also use `pydantic.mypy`, **`sqlmodel_mypy.plugin` must be listed first**:

```ini
[mypy]
plugins = sqlmodel_mypy.plugin, pydantic.mypy
```

This is not a preference: **mypy picks the first plugin that claims a hook**. If `pydantic.mypy` runs first, it
can treat SQLModel classes as plain Pydantic models and you’ll get incorrect typing (e.g. missing `__table__`,
broken SQLAlchemy expression types, and incorrect/empty constructor signatures).

## Plugin configuration

All options are **boolean**.

Supported options:

- **`init_typed`** (default: `false`): Generate `__init__` / constructor signatures using the declared field types.
  When `false`, the plugin uses `Any` for parameter types (but still enforces required/optional fields).
- **`init_forbid_extra`** (default: `false`): When `true`, do **not** add `**kwargs: Any` to generated
  `__init__` / `model_construct` / constructor signatures, so mypy reports unexpected keyword arguments.
- **`warn_untyped_fields`** (default: `true`): When `true`, emit error code `sqlmodel-field` for untyped
  declarations like `x = Field(...)` / `x = Relationship(...)` (use `x: T = ...` instead).
- **`debug_dataclass_transform`** (default: `false`): **Advanced/debug-only**. When `true`, keep SQLModel’s
  `__dataclass_transform__` handling enabled in mypy (useful for debugging plugin interactions; not recommended
  for normal use).

`mypy.ini`:

```ini
[sqlmodel-mypy]
init_typed = false
init_forbid_extra = false
warn_untyped_fields = true
debug_dataclass_transform = false
```

`pyproject.toml`:

```toml
[tool.sqlmodel-mypy]
init_typed = false
init_forbid_extra = false
warn_untyped_fields = true
debug_dataclass_transform = false
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

You can also use SQLAlchemy table metadata attributes without `attr-defined` noise:

```py
tbl = User.__table__
```

### `getattr(Model, "field")` support

If you build query filters dynamically, `getattr(Model, "field")` with a **string literal** is typed the same as
direct `Model.field` access on `table=True` models:

```py
stmt = select(User).where(getattr(User, "name").like("%x%"))
```

Non-literal names (runtime strings) are intentionally left as `Any` (you’ll still need a cast or a different
pattern).

### Relationship comparator typing

Relationship attributes declared via `Relationship(...)` are typed as SQLAlchemy expressions at class level,
including common relationship comparators used in query filters:

```py
stmt = select(Team).where(Team.heroes.any(Hero.name == "x"))
stmt = select(Hero).where(Hero.team.has(Team.name == "t"))
stmt = select(Team).where(Team.heroes.contains(hero_obj))
```

### Out of scope

- **Most SQLAlchemy-only declaration helpers**: this plugin doesn’t try to “fix” the full *declaration-site* typing
  for SQLAlchemy ORM APIs that aren’t part of SQLModel’s surface area.
  - Exception: best-effort return typing for `column_property(...)` (infers `Mapped[T]` from `ScalarSelect[T]` /
    `ColumnElement[T]`) to support patterns like `Model._foo = column_property(...)` combined with
    `getattr(Model, "_foo")` in query builders.
- **Pydantic `@computed_field`**: not touched by this plugin (it should type-check via normal property typing / Pydantic
  typing).

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

## Persisted model helpers (opt-in)

Strict typing + SQLModel often means table-model IDs start as `None` and are populated after flush/commit/refresh
(see [SQLModel: Automatic IDs, None, and Refresh](https://sqlmodel.tiangolo.com/tutorial/automatic-id-none-refresh/)).
Mypy can’t infer those runtime effects, so you can end up with repeated `assert obj.id is not None`.

This package ships small helpers you can import (they do **not** affect plugin behavior):

```py
from sqlmodel_mypy import has_id, require_id

hero_id: int = require_id(hero)  # raises ValueError if hero.id is None

if has_id(hero):
    ok_id: int = hero.id
```

## Development

- List available commands: `make help`
- Lint/format: `make lint` / `make fmt`
- Typecheck plugin code: `make typecheck`
- Run tests: `make test`
- Run all quality gates: `make check`
- Update mypy "golden" outputs: `make update-mypy`

See [`AGENTS.md`](AGENTS.md) for repo conventions and workflow notes.

## Security

See [`SECURITY.md`](SECURITY.md).

## Code of conduct

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

# Roadmap

## v0.1 (MVP) - DONE

- Correct required/optional kwargs for `sqlmodel.Field(...)` when generating `__init__`.
- Exclude `sqlmodel.Relationship(...)` from generated `__init__` kwargs.
- Minimal plugin config + cache invalidation via `report_config_data()`.
- Mypy integration test suite with golden outputs.

## v0.2 - DONE

- Config parity with Pydantic plugin (more knobs; documented):
  - `init_typed`
  - `init_forbid_extra`
- Better handling of inherited fields (including generics/typevars) and overrides.

## v0.3 - DONE

- Relationship typing improvements (forward refs, `Mapped[...]` wrapping behavior).
- Typed SQL expression building (reduce need for `col()`, `table.c.*`, etc.):
  - Improve class/instance attribute typing so `User.id`, `User.name`, etc. are well-typed in expressions.
  - Ensure common SQLModel re-exports (`select`, `and_`, `or_`, `col`, etc.) preserve SQLAlchemy typing.
  - Decide approach: plugin hooks vs stub overlays vs relying on SQLAlchemy typing (documented trade-offs).
- Compatibility matrix CI across mypy/SQLModel versions (and pin policy).
- **Plugin ordering guardrails (mypy limitation)**
  - **Reality check**: mypy generally uses the **first plugin that claims a hook**, so true “order independence”
    between `sqlmodel_mypy.plugin` and `pydantic.mypy` is not realistically achievable today.
  - Enforce and clearly explain that `sqlmodel_mypy.plugin` must be listed **before** `pydantic.mypy`:
    - Source: `src/sqlmodel_mypy/plugin.py` (`SQLModelMypyPlugin.__init__`, `_sqlmodel_metaclass_callback()`).
    - User-facing docs: `README.md` (plugin ordering section).

## v0.4 (Docs-aligned relationship constructors) - DONE

**Motivation**: SQLModel’s official docs pass relationship attributes in constructors (e.g. `Hero(team=team)`,
`Team(heroes=[...])`). v0.4 accepts relationship kwargs in generated `__init__` / `model_construct` for
`table=True` models, so `init_forbid_extra=true` no longer causes false-positive “unexpected keyword argument”
errors for documented patterns.

- **Accept relationship kwargs in generated `__init__` / `model_construct` (table models)**
  - Allow many-to-one and one-to-many patterns from the docs:
    - `Hero(..., team=team_obj)` / `hero.team = team_obj`
    - `Team(..., heroes=[hero1, hero2])` / `team.heroes.append(hero)`
    - Docs: `https://sqlmodel.tiangolo.com/tutorial/relationship-attributes/create-and-update-relationships/`
  - Allow many-to-many relationship kwargs (lists) from the docs:
    - `Hero(..., teams=[team1, team2])` with `link_model=...`
    - Docs: `https://sqlmodel.tiangolo.com/tutorial/many-to-many/create-data/`
  - Keep behavior **scoped to `table=True` models** (relationship kwargs are applied after construction in SQLModel):
    - Source: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/_compat.py` (`sqlmodel_table_construct()`,
      `sqlmodel_validate()` assign relationship keys after fields).
  - Add/extend mypy integration tests covering relationship kwargs under both configs:
    - `init_forbid_extra=false` and `init_forbid_extra=true`
    - `init_typed=false` and `init_typed=true`

- **Join/outer-join query typing coverage (SQLModel docs patterns)**
  - Ensure expression typing works for doc examples using joins:
    - `select(Hero, Team).join(Team)` and `select(Hero, Team).join(Team, isouter=True)`
    - Docs: `https://sqlmodel.tiangolo.com/tutorial/connect/read-connected-data/`

## v0.4.1 (Outer-join `None` propagation) - DONE

- Decide how to model **outer-join `None` propagation** in result tuples (docs show `Team: None`):
  - Either type `Team | None` on the right-hand side when `isouter=True`, or document a recommended narrowing
    pattern (and keep types conservative).
  - Update/add mypy integration tests to match the chosen strategy.

## v0.5 (Model-kind awareness + reduce “surprising” typing) - DONE

- **Table-model awareness for SQLAlchemy expression typing**
  - Only treat class attributes as SQLAlchemy expressions (e.g. wrap to `InstrumentedAttribute[T]`) for `table=True`
    SQLModel models.
  - Avoid changing attribute types for “data models” (non-table SQLModel subclasses) used for FastAPI schemas, e.g.
    `TeamBase`, `HeroCreate`, `HeroPublic`.
    - Docs: `https://sqlmodel.tiangolo.com/tutorial/fastapi/relationships/`

- **Pydantic/Field alias & config parity checkpoints**
  - Verify our generated signatures stay compatible with SQLModel’s `Field(...)` alias behavior (alias propagation
    and precedence in `validation_alias` / `serialization_alias`).
    - Source: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/main.py` (`Field()` implementation).
  - Document what is supported vs intentionally out of scope when users rely on aliases in constructor kwargs.

## v0.6 (Session.exec / AsyncSession.exec typing gaps) - DONE

- **Broaden `Session.exec()` / `AsyncSession.exec()` accepted statement types**
  - Fix common mypy failures when passing non-`Select` statements (e.g. `delete(...)`) to `exec()`.
  - Implement via upstream PRs and/or stub overlays (keep runtime behavior unchanged):
    - Source: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/orm/session.py`
    - Source: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/ext/asyncio/session.py`
    - Known gap example: `https://github.com/fastapi/sqlmodel/discussions/831`

## v0.7 (Advanced SQL expression + relationship comparator typing) - DONE

- **Relationship comparator typing**
  - `.any()`, `.has()`, `.contains()`, etc. on relationship attributes in query expressions.
- **More query-shape coverage**
  - Multi-entity selects, joins/outer-joins, and relationship-based filters across common SQLModel tutorial patterns.
- **Document trade-offs (and pick a default strategy)**
  - Plugin hooks vs stub overlays vs relying purely on SQLAlchemy typing (with “when to use what” guidance).

## v0.8 (Field `sa_type=` supports SQLAlchemy TypeEngine instances) - DONE

**Motivation**: SQLModel’s `Field(sa_type=...)` is a supported API, but real-world usage commonly passes
**instantiated SQLAlchemy types** (e.g. `DateTime(timezone=True)`, `String(50)`), which triggers mypy
`call-overload` errors under `--strict`. Users currently work around this with `sa_column=Column(...)` or
`# type: ignore`, which is exactly the “types clutter” we want to remove.

- **Fix mypy `call-overload` for `Field(sa_type=<TypeEngine instance>)`**
  - Upstream issue reports:
    - `https://github.com/fastapi/sqlmodel/discussions/955`
    - `https://github.com/fastapi/sqlmodel/discussions/1228`
  - Upstream pending fix (adjust `sa_type` annotation to match SQLAlchemy’s `Column` type argument rules):
    - `https://github.com/fastapi/sqlmodel/pull/1345`
  - Source references:
    - SQLModel `Field()` signature: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/main.py`
    - SQLAlchemy `Column` accepts both a **TypeEngine class** and a **TypeEngine instance** (see PR #1345 write-up).

- **Implementation options (pick default; document trade-offs)**
  - **Preferred**: upstream-first — rely on SQLModel merging PR #1345; in our plugin add a compatibility layer for
    older SQLModel releases and/or for users stuck on versions without the fix.
  - **Plugin signature hook**: add a `Field()` signature hook that widens `sa_type` to accept
    `sqlalchemy.sql.type_api.TypeEngine[Any] | type[Any]` when present.
  - **Stub overlay**: ship a small `.pyi` overlay (or `typing`-only shim) to widen the `Field()` overload(s) without
    touching runtime behavior.

- **Tests**
  - Add a mypy integration module that reproduces the failure in strict mode and asserts it’s fixed without
    `# type: ignore`, e.g.:
    - `Field(sa_type=DateTime(timezone=True))`
    - `Field(sa_type=String(50))`
  - Ensure the fix is **idempotent**: if upstream SQLModel already accepts instances, our hook/overlay must not
    change behavior or introduce new ambiguities.

## v0.9 (`model_config = ConfigDict(...)` compatibility in strict mode) - DONE

**Motivation**: SQLModel is built on Pydantic v2, and users often want to set things like `extra="forbid"` on
schema-like models (e.g. `HeroCreate`, `HeroUpdate`) via `model_config = ConfigDict(...)`. Today this can fail mypy
strict with an “incompatible types in assignment” error (`ConfigDict` vs `SQLModelConfig`), forcing users into
`# type: ignore` or `cast(...)`.

- **Fix mypy incompatibility for `model_config` overrides**
  - Upstream discussion / reproduction:
    - `https://github.com/fastapi/sqlmodel/discussions/855`
  - Source references:
    - SQLModel config typing helpers: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/_compat.py`
    - SQLModel base class `model_config` annotation: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/main.py`

- **Implementation options (pick default; document trade-offs)**
  - **Stub overlay (likely simplest)**: type `SQLModel.model_config` as `pydantic.ConfigDict` (or `Mapping[str, Any]`)
    so assignment from `ConfigDict(...)` is always accepted.
  - **Plugin adjustment**: during semantic analysis, rewrite the inferred type of `model_config` on SQLModel subclasses
    to a compatible supertype (while keeping runtime untouched).
  - Ensure we don’t regress table-model behavior: SQLModel uses `model_config["table"]` and `model_config["registry"]`
    at runtime (see `SQLModelMetaclass.__new__`).

- **Tests**
  - Add a mypy integration module proving strict-mode acceptance without ignores, e.g.:
    - `from pydantic import ConfigDict`
    - `class HeroCreate(SQLModel): model_config = ConfigDict(extra="forbid")`
  - Include a minimal table model + data model inheritance case (common in SQLModel docs) to ensure no false positives.

## v0.10 (`select()` varargs: avoid overload ceiling) - DONE

**Motivation**: SQLModel provides its own typed `select()` overloads, but they are generated only up to **4**
entities. In real code it’s common to select more than 4 columns/entities (especially with joins/aggregates), which
can trigger “No overload variant of `select` matches argument types …” even though runtime works.

- **Fix overload ceiling for `sqlmodel.select(...)`**
  - Source reference (generated overloads; note the 4-entity cap):
    - `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/sql/_expression_select_gen.py`
  - Upstream issue reports:
    - `https://github.com/fastapi/sqlmodel/issues/92`
    - `https://github.com/fastapi/sqlmodel/issues/271`

- **Implementation options (pick default; document trade-offs)**
  - **Stub overlay (likely simplest)**: add a final “varargs fallback” overload, e.g.:
    - `@overload`
    - `def select(__ent0: Any, __ent1: Any, __ent2: Any, __ent3: Any, __ent4: Any, *entities: Any) -> Select[tuple[Any, ...]]: ...`
    so mypy always has a matching overload beyond 4 args.
  - **Plugin signature hook**: provide a mypy hook for `sqlmodel.sql.expression.select` that adds the same fallback
    overload, keeping the existing 1–4 entity overload precision.
  - Optionally, generate more precise overloads (e.g. up to 8) if we’re willing to carry the maintenance cost.

- **Tests**
  - Add a mypy integration module that selects 5+ entities and asserts no `call-overload` errors under strict mode.
  - Decide and lock down the fallback return type (likely `Select[tuple[Any, ...]]`) to keep behavior predictable.

## v0.11 (SQLAlchemy “extras” coverage + SQLModel `Select` join parity) - DONE

- **Lock down common SQLAlchemy helpers used with SQLModel**
  - `.label(...)` on `table=True` model fields typed as SQLAlchemy expressions.
  - ORM loader options like `selectinload(Model.relationship)` on `table=True` relationships.
  - `Select.execution_options(...)` chaining without losing the `Select[...]` generic (including outer-join `None` propagation).
- **Fix outer-join `None` propagation for SQLModel’s own `Select` wrapper**
  - Apply the join/outerjoin return-type adjustment to both SQLAlchemy `Select` and SQLModel’s `sqlmodel.sql._expression_select_cls.Select`.
- **Tests**
  - Add/extend mypy integration coverage and commit golden outputs for the new cases.

## v0.12 (`__table__` dunder typing + `Session.get`/`AsyncSession.get` coverage) - DONE

**Motivation**: SQLAlchemy APIs and patterns commonly rely on ORM models exposing `Model.__table__`. SQLModel
models do provide this at runtime for `table=True`, but type checkers can miss it and report `attr-defined`.

- **Expose and type `__table__` on `table=True` SQLModel models**
  - Provide a plugin-generated `__table__` attribute typed as `sqlalchemy.sql.schema.Table` (with safe fallbacks
    when stubs don’t expose `Table`).
- **Tests**
  - Add mypy integration coverage for `Model.__table__` and `Session.get` / `AsyncSession.get` with SQLModel
    table models.

## v0.12.4 (Dynamic query-builder helpers) - DONE

- Type `getattr(Model, "field")` like `Model.field` for `table=True` SQLModel models (string-literal names only).
- Best-effort typing for `column_property(...)` return type (infer `Mapped[T]`) to support
  `Model._foo = column_property(...)` patterns.

## v0.14 (`typing.Annotated[...]` field metadata support) - DONE

**Motivation**: SQLModel (Pydantic v2) supports `Annotated`-driven typing patterns, and users will naturally try to
use them to reduce `= Field(...)` assignment noise. Under `--strict`, we should still generate correct constructor
signatures (required/optional + alias kwargs) when `Field(...)` metadata lives inside `Annotated[...]`.

- **Support `Annotated[T, Field(...)]` for constructor/signature generation**
  - Examples we should handle (at least best-effort):
    - `id: Annotated[int | None, Field(default=None, primary_key=True)]`
    - `name: Annotated[str, Field(alias="full_name")]`
    - The historical SQLModel edge-case: `Optional[Annotated[T, ...]]` (see upstream release notes).
  - Implementation sketch:
    - Extend semantic-phase collection in `src/sqlmodel_mypy/transform.py` to extract Field metadata from the
      annotation when there is no `= Field(...)` call in the class body.
    - Mirror the same logic in checker-time signature collection in `src/sqlmodel_mypy/plugin.py`
      (`_collect_member_from_stmt`) so class-call signatures stay consistent with generated `__init__`.
  - Source references:
    - SQLModel `Field()` overloads + alias precedence: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/main.py`
    - `Annotated` handling in SQLModel’s type parsing: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/_compat.py`
    - Release note: “Fix support for types with `Optional[Annotated[x, f()]]`…” (PR #1093):
      `https://github.com/fastapi/sqlmodel/pull/1093`

## v0.15 (Outer-join `None` propagation for relationship-attribute joins) - DONE

**Motivation**: we currently propagate `None` on outer-joins only when the join target is a direct model class
(e.g. `.join(Team, isouter=True)`). A common ORM pattern is to join via relationship attributes
(e.g. `.join(Hero.team, isouter=True)`), and the result tuple should still reflect that the joined entity may be
`None`.

- **Expand join-target detection**
  - Extend `src/sqlmodel_mypy/plugin.py` (`_sqlalchemy_select_join_like_return_type`) to also detect join targets
    expressed as relationship attributes (e.g. `Hero.team`, `Team.heroes`) and map them to the target model type
    for `None` propagation.
  - References:
    - SQLModel outer-join behavior (baseline example): `https://sqlmodel.tiangolo.com/tutorial/connect/read-connected-data/`
    - SQLAlchemy relationship-attribute joins (`select(User).join(User.addresses)`): `https://docs.sqlalchemy.org/en/20/orm/queryguide/select.html#simple-relationship-joins`
  - Add mypy integration coverage under `tests/mypy/modules/` for relationship-attribute joins, including the SQLModel
    docs pattern of `Team: None` on outer-joins.

## v0.16 (Strict typing ergonomics add-ons, opt-in) - DONE

**Motivation**: strict typing + SQLModel often means the database populates values “later” (after flush/commit/
refresh), especially primary keys (`id: int | None`). Mypy can’t infer those runtime effects, and users end up with
repeated `assert obj.id is not None` (type clutter).

- **Provide a tiny opt-in narrowing helper**
  - Docs context (why `id` starts as `None` and is populated after commit/refresh):
    - `https://sqlmodel.tiangolo.com/tutorial/automatic-id-none-refresh/`
  - Ship a small helper in this package (runtime-free or near-zero runtime) that helps users narrow “persisted” models
    without `cast(...)`/`# type: ignore`, e.g.:
    - `assert_persisted(obj)` / `require_id(obj)` / `ensure_persisted(obj)` returning a `TypeGuard[...]` or using
      overloads to narrow `id` to `int`.
  - Keep it optional and well-documented; default plugin behavior should remain conservative.

## v0.17 (`tuple_()` should return a SQL expression, not a Python tuple)

**Motivation**: SQLModel re-exports `tuple_` from `sqlmodel.sql.expression`, but its typing currently says it returns a
built-in `tuple[Any, ...]`. In SQLAlchemy, `tuple_()` returns a SQL expression object (`sqlalchemy.sql.elements.Tuple`)
which supports expression methods like `.in_(...)`. Under `--strict` this mismatch forces `cast(...)` / `# type: ignore`
for otherwise normal SQLAlchemy patterns.

- **Fix `sqlmodel.tuple_` / `sqlmodel.sql.expression.tuple_` return type**
  - Add a plugin return-type hook for:
    - `sqlmodel.sql.expression.tuple_`
    - `sqlmodel.tuple_` (re-export)
  - Prefer returning `sqlalchemy.sql.elements.Tuple` (or fall back to `ColumnElement[tuple[Any, ...]]` if stubs vary).
  - Source references:
    - SQLModel wrapper: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/sql/expression.py`
    - SQLAlchemy `tuple_` constructor: `https://github.com/sqlalchemy/sqlalchemy/blob/main/lib/sqlalchemy/sql/_elements_constructors.py`
    - SQLAlchemy `Tuple` expression class: `https://github.com/sqlalchemy/sqlalchemy/blob/main/lib/sqlalchemy/sql/elements.py`
- **Tests**
  - Add a mypy integration module demonstrating composite IN typing without ignores, e.g.:
    - `tuple_(User.id, User.team_id).in_([(1, 2)])`

## v0.18 (Recognize `table=True` via `model_config["table"]`, not just class kwargs)

**Motivation**: SQLModel’s runtime `is_table_model_class()` checks `model_config.get("table")`, so a class can be a
table model even when it isn’t declared as `class Model(SQLModel, table=True)`. Today our plugin/transformer only
looks at the class keyword, so “table-only” behaviors may not activate (expression typing, relationship kwargs in
constructor signatures, `__table__`, etc.).

- **Detect table models via `model_config` (metadata-driven)**
  - Extend semantic analysis in `src/sqlmodel_mypy/transform.py` to detect a statically-known class-body assignment like:
    - `model_config = ConfigDict(table=True)` or equivalent SQLModel config helpers
  - Persist the result in `TypeInfo.metadata["sqlmodel-mypy-metadata"]`, then update `_is_table_model(...)` in:
    - `src/sqlmodel_mypy/plugin.py`
    - `src/sqlmodel_mypy/transform.py`
    to consult metadata first (then fall back to `table=True` in the class definition).
  - Source references:
    - SQLModel metaclass reads config: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/main.py` (`SQLModelMetaclass.__new__`)
    - SQLModel runtime table detection: `https://github.com/fastapi/sqlmodel/blob/main/sqlmodel/_compat.py` (`is_table_model_class`)
- **Tests**
  - Add mypy integration coverage for a table model enabled via `model_config` and assert the same behavior as
    `table=True`:
    - class attribute expression typing (e.g. `User.id.in_([1])`)
    - constructor accepts relationship kwargs when configured (`init_forbid_extra=true`)
    - `User.__table__` is typed and present

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
- Avoid forcing users to order plugins (`sqlmodel_mypy.plugin` vs `pydantic.mypy`) for correct SQLModel resolution.

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

## v0.7+ (Advanced SQL expression + relationship comparator typing)

- **Relationship comparator typing**
  - `.any()`, `.has()`, `.contains()`, etc. on relationship attributes in query expressions.
- **More query-shape coverage**
  - Multi-entity selects, joins/outer-joins, and relationship-based filters across common SQLModel tutorial patterns.
- **Document trade-offs (and pick a default strategy)**
  - Plugin hooks vs stub overlays vs relying purely on SQLAlchemy typing (with “when to use what” guidance).

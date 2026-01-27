# Roadmap

## v0.1 (MVP)

- Correct required/optional kwargs for `sqlmodel.Field(...)` when generating `__init__`.
- Exclude `sqlmodel.Relationship(...)` from generated `__init__` kwargs.
- Minimal plugin config + cache invalidation via `report_config_data()`.
- Mypy integration test suite with golden outputs.

## v0.2

- Config parity with Pydantic plugin (more knobs; documented):
  - `init_typed`
  - `init_forbid_extra`
- Better handling of inherited fields (including generics/typevars) and overrides.

## v0.3+

- Relationship typing improvements (forward refs, `Mapped[...]` wrapping behavior).
- Typed SQL expression building (reduce need for `col()`, `table.c.*`, etc.):
  - Improve class/instance attribute typing so `User.id`, `User.name`, etc. are well-typed in expressions.
  - Ensure common SQLModel re-exports (`select`, `and_`, `or_`, `col`, etc.) preserve SQLAlchemy typing.
  - Decide approach: plugin hooks vs stub overlays vs relying on SQLAlchemy typing (documented trade-offs).
- Compatibility matrix CI across mypy/SQLModel versions (and pin policy).

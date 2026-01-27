## Summary

Explain the problem and the approach.

## Test plan

- [ ] `make check` (required)
- [ ] Coverage 90% (enforced by `make test` / `make check`)
- [ ] If mypy output/signatures changed: ran `make update-mypy` and committed updated `tests/mypy/outputs/**`
- [ ] Added/updated tests as needed

## Notes for maintainers

- [ ] If plugin behavior changed: bumped `src/sqlmodel_mypy/plugin.py::__version__` (mypy cache invalidation)

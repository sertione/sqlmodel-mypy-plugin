from __future__ import annotations

import pytest

from sqlmodel_mypy.persisted import has_id, require_id


class Obj:
    def __init__(self, id: int | None) -> None:
        self.id = id


def test_has_id() -> None:
    assert has_id(Obj(1)) is True
    assert has_id(Obj(None)) is False


def test_require_id_returns_value() -> None:
    assert require_id(Obj(1)) == 1


def test_require_id_raises_on_none() -> None:
    with pytest.raises(ValueError, match="non-None `id`"):
        require_id(Obj(None))

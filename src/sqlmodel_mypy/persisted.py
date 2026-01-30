from __future__ import annotations

from typing import Protocol, TypeVar

from typing_extensions import TypeIs

TId_co = TypeVar("TId_co", covariant=True)


class HasOptionalId(Protocol[TId_co]):
    """An object with an optional primary key (populated later by the DB)."""

    @property
    def id(self) -> TId_co | None: ...


class HasId(HasOptionalId[TId_co], Protocol[TId_co]):
    """An object with a non-optional primary key."""

    @property
    def id(self) -> TId_co: ...


def has_id(obj: HasOptionalId[TId_co]) -> TypeIs[HasId[TId_co]]:
    """Return True if `obj.id` is not None, and narrow `obj.id` accordingly."""

    return obj.id is not None


def require_id(obj: HasOptionalId[TId_co]) -> TId_co:
    """Return `obj.id` if present; raise if it's None."""

    if obj.id is None:
        raise ValueError("Expected object to have a non-None `id` (persisted object).")
    return obj.id

from __future__ import annotations

from sqlalchemy.sql.expression import ColumnElement
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field()


# `__table__` should exist on table models (and should not require importing Table explicitly).
_ = User.__table__


# Class attribute access should be typed as SQLAlchemy expressions.
predicates: list[ColumnElement[bool]] = []
predicates.append(User.id == 1)
predicates.append(User.id.in_([1, 2]))
predicates.append(User.name.like("%x%"))

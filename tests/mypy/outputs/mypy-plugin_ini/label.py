from __future__ import annotations

from sqlmodel import Field, SQLModel, select


class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field()


stmt = select(User.id.label("user_id"))

# `label()` returns an SQL expression, not a plain int.
bad_int: int = User.id.label("user_id")
# MYPY: error: Incompatible types in assignment (expression has type "Label[int]", variable has type "int")  [assignment]

from __future__ import annotations

from sqlmodel import Field, SQLModel, and_, select


class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field()


stmt = select(User).where(and_(User.id.in_([1, 2]), User.name.like("%x%")))

# Class attribute access should be an SQL expression, not a plain value.
bad_int: int = User.id

# SQL expressions are not plain bools.
bad_bool: bool = User.name.like("%x%")

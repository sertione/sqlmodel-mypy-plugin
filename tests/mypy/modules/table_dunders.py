from __future__ import annotations

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import InstanceState, Mapper
from sqlalchemy.sql.schema import Table
from sqlmodel import Field, SQLModel
from sqlmodel import inspect as sm_inspect


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


ok_table: Table = User.__table__
ok_mapper: Mapper[User] = User.__mapper__

# `inspect(...)` should be typed for SQLModel table models (SQLAlchemy + SQLModel re-export).
ok_mapper2: Mapper[User] = sa_inspect(User)
ok_mapper3: Mapper[User] = sm_inspect(User)

user = User(name="x")
ok_state: InstanceState[User] = sa_inspect(user)
ok_state2: InstanceState[User] = sm_inspect(user)

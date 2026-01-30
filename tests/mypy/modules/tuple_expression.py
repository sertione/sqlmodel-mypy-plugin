from __future__ import annotations

from sqlmodel import Field, SQLModel, tuple_
from sqlmodel.sql import expression as sql_expression


class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    team_id: int = Field()


# `tuple_()` should return a SQL expression (supports `.in_(...)`), not a Python tuple.
tuple_(User.id, User.team_id).in_([(1, 2)])
sql_expression.tuple_(User.id, User.team_id).in_([(1, 2)])

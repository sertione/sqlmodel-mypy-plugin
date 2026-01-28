from __future__ import annotations

from sqlalchemy.sql.schema import Table
from sqlmodel import Field, SQLModel


class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


ok_table: Table = User.__table__

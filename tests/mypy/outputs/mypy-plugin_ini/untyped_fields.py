from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel):
    name = Field()
# MYPY: error: Untyped fields disallowed  [sqlmodel-field]


class Hero(SQLModel):
    team_id: int | None = Field(default=None)
    team = Relationship(back_populates="heroes")
# MYPY: error: Untyped fields disallowed  [sqlmodel-field]

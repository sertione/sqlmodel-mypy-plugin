from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel):
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel):
    name: str = Field()
    team: Team | None = Relationship(back_populates="heroes")


Hero(team=None)
# MYPY: error: Missing named argument "name" for "Hero"  [call-arg]
Hero(name="x", team=None)

from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel):
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel):
    name: str = Field()
    team: Team | None = Relationship(back_populates="heroes")


Hero(team=None)
Hero(name="x", team=None)

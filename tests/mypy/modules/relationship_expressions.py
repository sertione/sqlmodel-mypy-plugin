from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel, table=True):
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    name: str = Field()
    team: Team | None = Relationship(back_populates="heroes")


# Relationship class attributes should be SQLAlchemy instrumented attributes, not plain values.
bad_heroes: list[Hero] = Team.heroes
bad_team: Team | None = Hero.team

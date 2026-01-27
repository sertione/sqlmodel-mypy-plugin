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
# MYPY: error: Incompatible types in assignment (expression has type "InstrumentedAttribute[list[Hero]]", variable has type "list[Hero]")  [assignment]
bad_team: Team | None = Hero.team
# MYPY: error: Incompatible types in assignment (expression has type "InstrumentedAttribute[Team | None]", variable has type "Team | None")  [assignment]

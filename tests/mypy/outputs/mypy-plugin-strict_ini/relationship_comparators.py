from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


# Comparator methods on relationship attributes should return SQL expressions,
# not plain bools / Any.
bad_bool_any: bool = Team.heroes.any()
# MYPY: error: Incompatible types in assignment (expression has type "ColumnElement[bool]", variable has type "bool")  [assignment]
bad_bool_any_crit: bool = Team.heroes.any(Hero.name == "x")
# MYPY: error: Incompatible types in assignment (expression has type "ColumnElement[bool]", variable has type "bool")  [assignment]
bad_bool_has: bool = Hero.team.has(Team.name == "t")
# MYPY: error: Incompatible types in assignment (expression has type "ColumnElement[bool]", variable has type "bool")  [assignment]

hero = Hero(name="h")
bad_bool_contains: bool = Team.heroes.contains(hero)
# MYPY: error: Incompatible types in assignment (expression has type "ColumnElement[bool]", variable has type "bool")  [assignment]


# Query-shape coverage (doc-like patterns).
stmt1 = select(Team).where(Team.heroes.any())
stmt2 = select(Team).where(Team.heroes.any(Hero.name == "x"))
stmt3 = select(Hero).where(Hero.team.has(Team.name == "t"))

# Multi-entity select + join + relationship-based filter.
stmt4 = select(Hero, Team).join(Team).where(Team.heroes.any(Hero.name == "x"))

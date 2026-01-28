from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class HeroTeamLink(SQLModel, table=True):
    team_id: int | None = Field(default=None, foreign_key="team.id", primary_key=True)
    hero_id: int | None = Field(default=None, foreign_key="hero.id", primary_key=True)


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="teams", link_model=HeroTeamLink)


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    teams: list[Team] = Relationship(back_populates="heroes", link_model=HeroTeamLink)


Hero(name="h", teams=[])
Hero(name="h", teams=[Team(name="t")])
Team(name="t", heroes=[Hero(name="h")])

# Type errors only when init_typed=true
Hero(name="h", teams=[1])
Team(name="t", heroes=[1])

# model_construct is always typed (independent of init_typed)
Hero.model_construct(name="h", teams=[])
Hero.model_construct(name="h", teams=[Team(name="t")])
Hero.model_construct(name="h", teams=[1])
# MYPY: error: List item 0 has incompatible type "int"; expected "Team"  [list-item]

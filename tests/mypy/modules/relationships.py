from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel, table=True):
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    name: str = Field()
    team: Team | None = Relationship(back_populates="heroes")


Hero(team=None)
Hero(name="x", team=None)

Team(name="t", heroes=[])

# Type errors only when init_typed=true
Hero(name="x", team=1)
Team(name="t", heroes=[1])

# model_construct is always typed (independent of init_typed)
Hero.model_construct(name="x", team=None)
Hero.model_construct(name="x", team=1)
Team.model_construct(name="t", heroes=[])
Team.model_construct(name="t", heroes=[1])

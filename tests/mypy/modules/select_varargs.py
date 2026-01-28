from __future__ import annotations

from sqlmodel import Field, SQLModel, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")


# SQLModel's generated `select()` overloads cap at 4 entities; ensure 5+ works.
stmt = select(Hero, Team, Hero.id, Team.id, Hero.name)

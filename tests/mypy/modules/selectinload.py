from __future__ import annotations

from sqlalchemy.orm import selectinload
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


stmt_team = select(Team).options(selectinload(Team.heroes))
stmt_hero = select(Hero).options(selectinload(Hero.team))

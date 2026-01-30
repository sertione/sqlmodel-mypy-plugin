from __future__ import annotations

from pydantic import ConfigDict
from sqlalchemy.sql.schema import Table
from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel):
    model_config = ConfigDict(table=True)

    id: int | None = Field(default=None, primary_key=True)
    name: str
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel):
    model_config = ConfigDict(table=True)

    id: int | None = Field(default=None, primary_key=True)
    name: str
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


# Expression typing should activate for table models declared via model_config.
ok_in = Hero.id.in_([1])

# `__table__` should be present/typed for table models declared via model_config.
ok_table: Table = Hero.__table__

# Relationship kwargs should be accepted for table models under init_forbid_extra=true.
team = Team(name="Team")
hero = Hero(name="Hero", team=team)
team2 = Team(name="Team2", heroes=[hero])
ok_team2: Team = team2

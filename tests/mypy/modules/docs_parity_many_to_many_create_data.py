from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class HeroTeamLink(SQLModel, table=True):
    team_id: int | None = Field(default=None, foreign_key="team.id", primary_key=True)
    hero_id: int | None = Field(default=None, foreign_key="hero.id", primary_key=True)


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    headquarters: str
    heroes: list[Hero] = Relationship(back_populates="teams", link_model=HeroTeamLink)


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    secret_name: str
    age: int | None = Field(default=None, index=True)
    teams: list[Team] = Relationship(back_populates="heroes", link_model=HeroTeamLink)


# Docs pattern: create many-to-many data via relationship attributes (lists).
team_preventers = Team(name="Preventers", headquarters="Sharp Tower")
team_z_force = Team(name="Z-Force", headquarters="Sister Margaret's Bar")

hero_deadpond = Hero(
    name="Deadpond",
    secret_name="Dive Wilson",
    teams=[team_z_force, team_preventers],
)
hero_rusty_man = Hero(
    name="Rusty-Man",
    secret_name="Tommy Sharp",
    age=48,
    teams=[team_preventers],
)
hero_spider_boy = Hero(
    name="Spider-Boy",
    secret_name="Pedro Parqueador",
    teams=[team_preventers],
)

ok_deadpond_teams: list[Team] = hero_deadpond.teams
ok_preventers_heroes: list[Hero] = team_preventers.heroes

# Intentional type error: teams list items must be Team.
Hero(name="Bad", secret_name="Bad", teams=[1])

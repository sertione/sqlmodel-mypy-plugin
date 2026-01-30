from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    headquarters: str
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    secret_name: str
    age: int | None = Field(default=None, index=True)
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


# Docs pattern: create instances with relationship attributes in constructors.
team_preventers = Team(name="Preventers", headquarters="Sharp Tower")
team_z_force = Team(name="Z-Force", headquarters="Sister Margaret's Bar")

hero_deadpond = Hero(name="Deadpond", secret_name="Dive Wilson", team=team_z_force)
hero_rusty_man = Hero(name="Rusty-Man", secret_name="Tommy Sharp", age=48, team=team_preventers)
hero_spider_boy = Hero(name="Spider-Boy", secret_name="Pedro Parqueador")

# Docs pattern: assign a relationship after creation.
hero_spider_boy.team = team_preventers

# Docs pattern: create a team with heroes in a list.
hero_black_lion = Hero(name="Black Lion", secret_name="Trevor Challa", age=35)
hero_sure_e = Hero(name="Princess Sure-E", secret_name="Sure-E")
team_wakaland = Team(
    name="Wakaland",
    headquarters="Wakaland Capital City",
    heroes=[hero_black_lion, hero_sure_e],
)
ok_team_wakaland: Team = team_wakaland

# Docs pattern: append to the relationship list.
hero_tarantula = Hero(name="Tarantula", secret_name="Natalia Roman-on", age=32)
team_preventers.heroes.append(hero_tarantula)

# Intentional type error: relationship kwargs must be Team|None, not int.
Hero(name="Bad", secret_name="Bad", team=1)
# MYPY: error: Argument "team" to "__init__" of "Hero" has incompatible type "int"; expected "Team | None"  [arg-type]

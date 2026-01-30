from __future__ import annotations

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlmodel.sql._expression_select_cls import Select as SQLModelSelect


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")


# SQLModel's generated `select()` overloads cap at 4 entities; ensure 5+ works.
stmt = select(Hero, Team, Hero.id, Team.id, Hero.name)

# Prefer returning SQLModel's own `Select[...]` wrapper, with best-effort element typing.
ok_stmt: SQLModelSelect[tuple[Hero, Team, int | None, int | None, str]] = stmt
# MYPY: error: Incompatible types in assignment (expression has type "Select[tuple[type[Hero], type[Team], int | None, int | None, str]]", variable has type "Select[tuple[Hero, Team, int | None, int | None, str]]")  [assignment]

engine = create_engine("sqlite://")

with Session(engine) as session:
    for hero, team, hero_id, team_id, hero_name in session.exec(stmt):
        ok_hero: Hero = hero
        ok_team: Team = team
        ok_team_id: int | None = team_id
        ok_hero_name: str = hero_name

        # `Hero.id` is optional at the type level.
        bad_hero_id: int = hero_id
        if hero_id is not None:
            ok_hero_id: int = hero_id

from __future__ import annotations

from sqlalchemy.orm import aliased
from sqlmodel import Field, Session, SQLModel, create_engine, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")


engine = create_engine("sqlite://")

team_alias = aliased(Team)

# Join targets expressed as variables (e.g. `aliased(Team)`) should still propagate `None` on outer joins.
stmt = select(Hero, team_alias).join(team_alias, isouter=True)

with Session(engine) as session:
    for hero, team in session.exec(stmt):
        ok_hero: Hero = hero
        ok_team: Team | None = team

        bad_team: Team = team
# MYPY: error: Incompatible types in assignment (expression has type "Team | None", variable has type "Team")  [assignment]
        if team is not None:
            ok_team_name: str = team.name

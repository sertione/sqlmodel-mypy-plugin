from __future__ import annotations

from sqlalchemy import select as sa_select
from sqlmodel import Field, Session, SQLModel, create_engine, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    team_id: int | None = Field(default=None, foreign_key="team.id")


engine = create_engine("sqlite://")
# The 5+ entity fallback and outer-join nullability must compose.
statement = select(Hero, Team, Hero.id, Team.id, Hero.name).outerjoin(Team)
with Session(engine) as session:
    for hero, team, hero_id, _team_id, name in session.exec(statement):
        ok_hero: Hero = hero
        ok_team: Team | None = team
        ok_id: int | None = hero_id
        ok_name: str = name
        bad_team: Team = team
        bad_id: int = hero_id
        bad_name: int = name

# A SQLModel select must preserve column arity when executed on a Core connection.
with engine.connect() as connection:
    for hero_id2, name2, _team_id2, _team_name2, _repeated_id in connection.execute(
        select(Hero.id, Hero.name, Hero.team_id, Team.name, Hero.id)
    ):
        ok_name2: str = name2
        bad_id2: int = hero_id2
        bad_name2: int = name2
    for hero3, team3 in connection.execute(sa_select(Hero, Team).outerjoin(Team)):
        ok_hero3: Hero = hero3
        bad_team3: Team = team3

from __future__ import annotations

from sqlmodel import Field, Session, SQLModel, create_engine, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")


engine = create_engine("sqlite://")

with Session(engine) as session:
    stmt_inner = select(Hero, Team).join(Team)
    for hero, team in session.exec(stmt_inner):
        ok_hero: Hero = hero
        ok_team: Team = team
        ok_team_name: str = team.name

    stmt_outer = select(Hero, Team).join(Team, isouter=True)
    for hero2, team2 in session.exec(stmt_outer):
        ok_hero2: Hero = hero2
        ok_team2: Team | None = team2

        # Outer join may yield `None` for the right-hand entity.
        bad_team: Team = team2
        if team2 is not None:
            ok_team_name2: str = team2.name

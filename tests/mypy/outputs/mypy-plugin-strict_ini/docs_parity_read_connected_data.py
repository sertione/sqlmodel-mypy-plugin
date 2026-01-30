from __future__ import annotations

from sqlmodel import Field, Session, SQLModel, create_engine, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    headquarters: str


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    secret_name: str
    age: int | None = Field(default=None, index=True)
    team_id: int | None = Field(default=None, foreign_key="team.id")


engine = create_engine("sqlite://")

with Session(engine) as session:
    # Docs pattern: select connected data with a WHERE clause (no join helper).
    stmt_where = select(Hero, Team).where(Hero.team_id == Team.id)
    for hero, team in session.exec(stmt_where):
        ok_hero: Hero = hero
        ok_team: Team = team
        ok_team_name: str = team.name

    # Docs pattern: select connected data using JOIN (with filter).
    stmt_join = select(Hero, Team).join(Team).where(Team.name == "Preventers")
    for hero2, team2 in session.exec(stmt_join):
        ok_hero2: Hero = hero2
        ok_team2: Team = team2

    # Docs pattern: select only heroes, but still JOIN for filtering.
    stmt_heroes_only = select(Hero).join(Team).where(Team.name == "Preventers")
    for hero3 in session.exec(stmt_heroes_only):
        ok_hero3: Hero = hero3

    # Docs pattern: LEFT OUTER join via isouter=True.
    stmt_outer = select(Hero, Team).join(Team, isouter=True)
    for hero4, team4 in session.exec(stmt_outer):
        ok_hero4: Hero = hero4
        ok_team4: Team | None = team4

        # Outer join may yield `None` for the right-hand entity.
        bad_team4: Team = team4
# MYPY: error: Incompatible types in assignment (expression has type "Team | None", variable has type "Team")  [assignment]

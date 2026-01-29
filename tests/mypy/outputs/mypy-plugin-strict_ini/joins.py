from __future__ import annotations

from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


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
# MYPY: error: Incompatible types in assignment (expression has type "Team | None", variable has type "Team")  [assignment]
        if team2 is not None:
            ok_team_name2: str = team2.name

    # Outer join via relationship attribute target should also propagate `None`.
    stmt_outer_rel = select(Hero, Team).join(Hero.team, isouter=True)
    for hero3, team3 in session.exec(stmt_outer_rel):
        ok_hero3: Hero = hero3
        ok_team3: Team | None = team3
        bad_team3: Team = team3
# MYPY: error: Incompatible types in assignment (expression has type "Team | None", variable has type "Team")  [assignment]

    # One-to-many join via relationship attribute target should also propagate `None`.
    stmt_outer_rel_many = select(Team, Hero).join(Team.heroes, isouter=True)
    for team4, hero4 in session.exec(stmt_outer_rel_many):
        ok_team4: Team = team4
        ok_hero4: Hero | None = hero4
        bad_hero4: Hero = hero4
# MYPY: error: Incompatible types in assignment (expression has type "Hero | None", variable has type "Hero")  [assignment]
        if hero4 is not None:
            ok_hero4_name: str = hero4.name

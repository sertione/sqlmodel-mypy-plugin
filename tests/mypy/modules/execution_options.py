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

stmt = select(Hero, Team).join(Team, isouter=True).execution_options(populate_existing=True)

# `execution_options()` should preserve the `Select[...]` generic, including outer-join `None`
# propagation for the joined entity.
ok_stmt = stmt

with Session(engine) as session:
    for hero, team in session.exec(stmt):
        ok_hero: Hero = hero
        ok_team: Team | None = team

        # Outer join may yield `None` for the right-hand entity.
        bad_team: Team = team

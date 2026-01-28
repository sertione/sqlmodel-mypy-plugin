from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel, table=True):
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    name: str = Field()
    team: Team | None = Relationship(back_populates="heroes")


Hero(team=None)
# MYPY: error: Missing named argument "name" for "__init__" of "Hero"  [call-arg]
Hero(name="x", team=None)

Team(name="t", heroes=[])

# Type errors only when init_typed=true
Hero(name="x", team=1)
# MYPY: error: Argument "team" to "__init__" of "Hero" has incompatible type "int"; expected "Team | None"  [arg-type]
Team(name="t", heroes=[1])
# MYPY: error: List item 0 has incompatible type "int"; expected "Hero"  [list-item]

# model_construct is always typed (independent of init_typed)
Hero.model_construct(name="x", team=None)
Hero.model_construct(name="x", team=1)
# MYPY: error: Argument "team" to "model_construct" of "Hero" has incompatible type "int"; expected "Team | None"  [arg-type]
Team.model_construct(name="t", heroes=[])
Team.model_construct(name="t", heroes=[1])
# MYPY: error: List item 0 has incompatible type "int"; expected "Hero"  [list-item]

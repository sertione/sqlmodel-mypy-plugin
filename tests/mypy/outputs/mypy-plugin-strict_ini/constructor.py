from __future__ import annotations

from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str

    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    secret_name: str = Field()
    age: int = 0
    team_id: int | None = Field(default=None, foreign_key="team.id")

    team: Team | None = Relationship(back_populates="heroes")


Team()
# MYPY: error: Missing named argument "name" for "__init__" of "Team"  [call-arg]
Team(name="Avengers")

Hero()
# MYPY: error: Missing named argument "name" for "__init__" of "Hero"  [call-arg]
# MYPY: error: Missing named argument "secret_name" for "__init__" of "Hero"  [call-arg]
Hero(name="Spiderman")
# MYPY: error: Missing named argument "secret_name" for "__init__" of "Hero"  [call-arg]
Hero(name="Spiderman", secret_name="Peter")
Hero(name="Spiderman", secret_name="Peter", extra=1)
# MYPY: error: Unexpected keyword argument "extra" for "__init__" of "Hero"  [call-arg]

# Type errors only when init_typed=true
Hero(name=1, secret_name=2)
# MYPY: error: Argument "name" to "__init__" of "Hero" has incompatible type "int"; expected "str"  [arg-type]
# MYPY: error: Argument "secret_name" to "__init__" of "Hero" has incompatible type "int"; expected "str"  [arg-type]

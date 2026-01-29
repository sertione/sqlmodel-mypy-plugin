# ruff: noqa: B009
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import column_property, selectinload
from sqlmodel import Field, Relationship, SQLModel, select


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    heroes: list[Hero] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


stmt_team = select(Team).options(selectinload(getattr(Team, "heroes")))

# `getattr(Model, "rel")` should be an ORM attribute (not the plain annotation type).
bad_heroes: list[Hero] = getattr(Team, "heroes")


class User(SQLModel, table=True):
    id: int = Field(primary_key=True)
    name: str = Field()


stmt_user = select(User).where(getattr(User, "name").like("%x%"))

# `getattr(Model, "field")` should be a SQL expression (not the plain value type).
bad_id: int = getattr(User, "id")


# Dynamic column_property assigned to a SQLModel class (common pattern for computed columns).
User._order_count = column_property(  # noqa: SLF001
    select(func.count(Hero.id)).where(Hero.team_id == Team.id).scalar_subquery()
)

stmt_count = select(User).where(getattr(User, "_order_count").between(1, 2))
bad_count: int = getattr(User, "_order_count")

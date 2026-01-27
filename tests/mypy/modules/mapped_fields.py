from __future__ import annotations

from sqlalchemy.orm import Mapped
from sqlmodel import Field, SQLModel


class User(SQLModel):
    id: Mapped[int] = Field(primary_key=True)
    name: str = Field()


User(id=1, name="x")

# `Mapped[T]` should be unwrapped to `T` in the constructor signature.
User(id=User.id, name="x")

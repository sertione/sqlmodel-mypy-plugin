from __future__ import annotations

from sqlmodel import Field, SQLModel


class Base(SQLModel):
    a: int = Field()
    b: int = Field(default=1)


class Child(Base):
    c: int = Field()
    b: int = Field(default=2)


Child()
Child(a=1, c=2)
Child(a=1)


class Base2(SQLModel):
    name: str = Field()


class BadOverride(Base2):
    def name(self) -> str:
        return "x"

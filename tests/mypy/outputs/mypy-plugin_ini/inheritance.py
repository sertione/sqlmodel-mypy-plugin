from __future__ import annotations

from sqlmodel import Field, SQLModel


class Base(SQLModel):
    a: int = Field()
    b: int = Field(default=1)


class Child(Base):
    c: int = Field()
    b: int = Field(default=2)


Child()
# MYPY: error: Missing named argument "a" for "__init__" of "Child"  [call-arg]
# MYPY: error: Missing named argument "c" for "__init__" of "Child"  [call-arg]
Child(a=1, c=2)
Child(a=1)
# MYPY: error: Missing named argument "c" for "__init__" of "Child"  [call-arg]


class Base2(SQLModel):
    name: str = Field()


class BadOverride(Base2):
    def name(self) -> str:
# MYPY: error: SQLModel field may only be overridden by another field  [sqlmodel-field]
# MYPY: error: Signature of "name" incompatible with supertype "Base2"  [override]
# MYPY: note:      Superclass:
# MYPY: note:          str
# MYPY: note:      Subclass:
# MYPY: note:          def name(self) -> str
        return "x"

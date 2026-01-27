from __future__ import annotations

from typing import Generic, TypeVar

from sqlmodel import Field, SQLModel

T_co = TypeVar("T_co", covariant=True)


class Box(SQLModel, Generic[T_co]):
    value: T_co = Field()


Box[int](value=1)
Box[int](value="x")
# MYPY: error: Argument "value" to "Box" has incompatible type "str"; expected "int"  [arg-type]

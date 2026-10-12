from __future__ import annotations

from typing import Generic, TypeAlias, TypeVar

from sqlmodel import SQLModel

JsonObject: TypeAlias = list["JsonObject"]
JsonValue: TypeAlias = str | int | None | list["JsonValue"] | dict[str, "JsonValue"]


class RecursiveTable(SQLModel, table=True):
    field: JsonObject


class Payload(SQLModel):
    value: JsonValue


RecursiveTable(field=[[], [[]]])
RecursiveTable(field="invalid")
RecursiveTable.model_construct(field=[[]])
RecursiveTable.model_construct(field=123)
Payload(value={"nested": ["text", 1, None]})
Payload(value=object())

T_co = TypeVar("T_co", covariant=True)
Values: TypeAlias = list[T_co]


class Box(SQLModel, Generic[T_co]):
    values: Values[T_co]


Box[int](values=[1])
Box[int](values=["invalid"])

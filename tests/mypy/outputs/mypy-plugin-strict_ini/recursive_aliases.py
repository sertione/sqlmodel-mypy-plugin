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
# MYPY: error: Argument "field" to "__init__" of "RecursiveTable" has incompatible type "str"; expected "JsonObject"  [arg-type]
RecursiveTable.model_construct(field=[[]])
RecursiveTable.model_construct(field=123)
# MYPY: error: Argument "field" to "model_construct" of "RecursiveTable" has incompatible type "int"; expected "JsonObject"  [arg-type]
Payload(value={"nested": ["text", 1, None]})
Payload(value=object())
# MYPY: error: Argument "value" to "__init__" of "Payload" has incompatible type "object"; expected "JsonValue"  [arg-type]

T_co = TypeVar("T_co", covariant=True)
Values: TypeAlias = list[T_co]


class Box(SQLModel, Generic[T_co]):
    values: Values[T_co]


Box[int](values=[1])
Box[int](values=["invalid"])
# MYPY: error: List item 0 has incompatible type "str"; expected "int"  [list-item]

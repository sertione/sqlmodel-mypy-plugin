from __future__ import annotations

from sqlmodel import SQLModel

type JsonObject = list[JsonObject]
type JsonValue = str | int | None | list[JsonValue] | dict[str, JsonValue]
type Left = list[Right]
type Right = dict[str, Left]


class RecursiveTable(SQLModel, table=True):
    field: JsonObject


class Payload(SQLModel):
    value: JsonValue
    mutual: Left


RecursiveTable(field=[[], [[]]])
RecursiveTable(field="invalid")
# MYPY: error: Argument "field" to "__init__" of "RecursiveTable" has incompatible type "str"; expected "JsonObject"  [arg-type]
RecursiveTable.model_construct(field=[[]])
RecursiveTable.model_construct(field=123)
# MYPY: error: Argument "field" to "model_construct" of "RecursiveTable" has incompatible type "int"; expected "JsonObject"  [arg-type]
Payload(value={"nested": ["text", 1, None]}, mutual=[{"child": []}])
Payload(value=object(), mutual=[])
# MYPY: error: Argument "value" to "__init__" of "Payload" has incompatible type "object"; expected "JsonValue"  [arg-type]
Payload(value=None, mutual=[123])
# MYPY: error: List item 0 has incompatible type "int"; expected "dict[str, Left]"  [list-item]

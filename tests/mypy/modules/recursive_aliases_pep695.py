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
RecursiveTable.model_construct(field=[[]])
RecursiveTable.model_construct(field=123)
Payload(value={"nested": ["text", 1, None]}, mutual=[{"child": []}])
Payload(value=object(), mutual=[])
Payload(value=None, mutual=[123])

from __future__ import annotations

from sqlmodel import Field, SQLModel


class DataModel(SQLModel):
    name: str = Field(alias="full_name")
    age: int = Field(default=0, alias="years")


class TableModel(SQLModel, table=True):
    name: str = Field(alias="full_name")
    age: int = Field(default=0, alias="years")


# Aliases should be accepted as constructor kwargs (including for required fields).
DataModel(full_name="x")
TableModel(full_name="x")

# Field names should still work.
DataModel(name="x")
TableModel(name="x")

# Optional field via alias.
DataModel(full_name="x", years=1)
TableModel(full_name="x", years=1)

# Type errors only when init_typed=true
DataModel(full_name=1)
TableModel(full_name=1)

# model_construct is always typed (independent of init_typed)
DataModel.model_construct(full_name="x")
DataModel.model_construct(full_name=1)
TableModel.model_construct(full_name="x")
TableModel.model_construct(full_name=1)


class OverrideValidationAlias(SQLModel):
    v: str = Field(alias="a", validation_alias="va")


OverrideValidationAlias(va="x")
OverrideValidationAlias(a="x")
OverrideValidationAlias.model_construct(va="x")
OverrideValidationAlias.model_construct(a="x")
OverrideValidationAlias.model_construct(va=1)


class SchemaExtraValidationAlias(SQLModel):
    x: str = Field(alias="a", schema_extra={"validation_alias": "sa"})


SchemaExtraValidationAlias(sa="x")
SchemaExtraValidationAlias(a="x")
SchemaExtraValidationAlias.model_construct(sa="x")
SchemaExtraValidationAlias.model_construct(sa=1)

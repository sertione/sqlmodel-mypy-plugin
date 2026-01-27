from __future__ import annotations

from sqlmodel import Field, SQLModel


class Model(SQLModel):
    name: str = Field()
    age: int = Field(default=0)


Model.model_construct(name="x")
Model.model_construct(name=1)
Model.model_construct(name="x", extra=1)
Model.model_construct(_fields_set={"name"}, name="x")
Model.model_construct(_fields_set={1}, name="x")

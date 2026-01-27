from __future__ import annotations

from sqlmodel import Field, SQLModel


class Model(SQLModel):
    name: str = Field()
    age: int = Field(default=0)


Model.model_construct(name="x")
Model.model_construct(name=1)
# MYPY: error: Argument "name" to "model_construct" of "Model" has incompatible type "int"; expected "str"  [arg-type]
Model.model_construct(name="x", extra=1)
# MYPY: error: Unexpected keyword argument "extra" for "model_construct" of "Model"  [call-arg]
Model.model_construct(_fields_set={"name"}, name="x")
Model.model_construct(_fields_set={1}, name="x")
# MYPY: error: Argument 1 to <set> has incompatible type "int"; expected "str"  [arg-type]

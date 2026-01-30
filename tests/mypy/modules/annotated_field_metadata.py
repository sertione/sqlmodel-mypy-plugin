from __future__ import annotations

from typing import Annotated, Optional

from sqlmodel import Field, SQLModel


class Data(SQLModel):
    id: Annotated[int | None, Field(default=None, primary_key=True)]
    name: Annotated[str, Field(alias="full_name")]


Data(full_name="x")
Data(name="x")

# Type errors only when init_typed=true
Data(full_name=1)

# model_construct is always typed (independent of init_typed)
Data.model_construct(full_name="x")
Data.model_construct(full_name=1)


class Opt(SQLModel):
    # Historical SQLModel edge-case: Optional[Annotated[T, Field(...)]]
    v: Optional[Annotated[str, Field(default=None, alias="a", validation_alias="va")]]  # noqa: UP045


Opt()
Opt(va="x")
Opt(a="x")
Opt(va=1)
Opt.model_construct(va="x")
Opt.model_construct(va=1)

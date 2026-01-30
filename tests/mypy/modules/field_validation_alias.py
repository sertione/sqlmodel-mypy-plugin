from __future__ import annotations

from pydantic.aliases import AliasChoices, AliasPath
from sqlmodel import Field, SQLModel


class Model(SQLModel):
    # Pydantic v2 supports non-string validation aliases; SQLModel's runtime accepts them too.
    x: int = Field(validation_alias=AliasPath("payload", "x"))
    y: int = Field(validation_alias=AliasChoices("y", AliasPath("payload", "y")))

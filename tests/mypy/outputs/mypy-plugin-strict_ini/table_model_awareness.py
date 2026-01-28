from __future__ import annotations

from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import Field, SQLModel


class UserBase(SQLModel):
    id: int = Field(primary_key=True)
    name: str = Field()


# Non-table models should behave like plain data models.
ok_id: int = UserBase.id
ok_name: str = UserBase.name

bad_instrumented: InstrumentedAttribute[int] = UserBase.id
# MYPY: error: Incompatible types in assignment (expression has type "int", variable has type "InstrumentedAttribute[int]")  [assignment]

bad_like = UserBase.name.like("%x%")
# MYPY: error: "str" has no attribute "like"  [attr-defined]

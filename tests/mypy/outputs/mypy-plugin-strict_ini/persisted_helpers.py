from __future__ import annotations

from sqlmodel import Field, SQLModel

from sqlmodel_mypy import has_id, require_id


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


hero = Hero(name="x")

# Narrowing helper should make `hero.id` non-optional inside the guarded block.
if has_id(hero):
    ok_id: int = hero.id

    # The narrowing should not discard other model attributes.
    ok_name: str = hero.name

# Outside the guard, `id` is still optional.
bad_id: int = hero.id
# MYPY: error: Incompatible types in assignment (expression has type "int | None", variable has type "int")  [assignment]


hero2 = Hero(id=1, name="y")
hero2_id: int = require_id(hero2)

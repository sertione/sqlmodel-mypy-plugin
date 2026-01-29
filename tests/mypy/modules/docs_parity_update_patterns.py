from __future__ import annotations

from sqlmodel import Field, SQLModel


class HeroBase(SQLModel):
    name: str = Field(index=True)
    secret_name: str
    age: int | None = Field(default=None, index=True)


class Hero(HeroBase, table=True):
    id: int | None = Field(default=None, primary_key=True)


class HeroCreate(HeroBase):
    pass


class HeroPublic(HeroBase):
    id: int


class HeroUpdate(SQLModel):
    # Docs pattern: optional update fields for PATCH-like updates.
    name: str | None = None
    secret_name: str | None = None
    age: int | None = None


# Docs pattern: dump only client-provided fields.
hero_update = HeroUpdate(name="Deadpuddle")
hero_data = hero_update.model_dump(exclude_unset=True)

# Docs pattern: apply patch data to an existing DB object.
db_hero = Hero(name="Deadpond", secret_name="Dive Wilson")
db_hero.sqlmodel_update(hero_data)

# Intentional type error: sqlmodel_update() only accepts dict or BaseModel.
db_hero.sqlmodel_update(1)

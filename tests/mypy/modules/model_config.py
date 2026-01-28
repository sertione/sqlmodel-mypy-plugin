from __future__ import annotations

from pydantic import ConfigDict
from sqlmodel import Field, SQLModel


class HeroBase(SQLModel):
    name: str


class HeroCreate(HeroBase):
    # Users commonly override `model_config` for schema-like models.
    model_config = ConfigDict(extra="forbid")


class Hero(HeroBase, table=True):
    # Common SQLModel docs pattern: data-model base + table model subclass.
    id: int | None = Field(default=None, primary_key=True)
    model_config = ConfigDict(extra="forbid")

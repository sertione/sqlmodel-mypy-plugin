from __future__ import annotations

import datetime as dt

from sqlalchemy import DateTime, String
from sqlmodel import Field, SQLModel


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: dt.datetime = Field(sa_type=DateTime(timezone=True))
    name: str = Field(sa_type=String(50))

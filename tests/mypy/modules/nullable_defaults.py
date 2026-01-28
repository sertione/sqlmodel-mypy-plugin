from __future__ import annotations

from sqlalchemy import Column, Computed, String, text
from sqlmodel import Field, SQLModel


class Model(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # Required field (no default, non-nullable).
    required: str = Field()

    # Nullable fields without explicit defaults should be optional constructor kwargs.
    note: str = Field(nullable=True)

    optional_sa: str = Field(sa_column=Column("optional_sa", String(), nullable=True))

    search_vector: str | None = Field(
        sa_column=Column("search_vector", String(), Computed("1", persisted=True))
    )

    created_by: str = Field(sa_column=Column("created_by", String(), server_default=text("'x'")))


Model()
Model(required="x")

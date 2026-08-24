from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import JSON, BigInteger, Column
from sqlmodel import Field, SQLModel


class Order(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # SQLAlchemy's `Sequence`, reached through a module alias: database-generated -> optional.
    number: int = Field(
        sa_column=Column(
            "number",
            BigInteger,
            sa.Sequence("example_number_seq", start=1, increment=1),
            index=True,
            nullable=False,
        )
    )

    # The bare name `Sequence` in this module is `collections.abc.Sequence`, not SQLAlchemy's.
    # It must not turn the field into an optional constructor kwarg.
    tags: Sequence[str] = Field(sa_column=Column("tags", JSON, nullable=False))


Order()
# MYPY: error: Missing named argument "tags" for "__init__" of "Order"  [call-arg]
Order(tags=["a"])

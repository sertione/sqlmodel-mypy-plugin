from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    FetchedValue,
    Identity,
    Integer,
    Sequence,
    String,
    text,
)
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
# MYPY: error: Missing named argument "required" for "__init__" of "Model"  [call-arg]
Model(required="x")


class GeneratedDefaults(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    required: str = Field()

    # Positional `Sequence(...)` becomes `Column.default`, exactly like `default=Sequence(...)`.
    seq_number: int = Field(
        sa_column=Column(
            "seq_number",
            BigInteger,
            Sequence("generated_defaults_seq_number_seq", start=1, increment=1),
            index=True,
            nullable=False,
        )
    )

    # Positional `Identity(...)` becomes `Column.server_default` (GENERATED ... AS IDENTITY).
    identity_number: int = Field(
        sa_column=Column("identity_number", Integer, Identity(always=True))
    )

    # Positional `FetchedValue()` becomes `Column.server_default`.
    fetched: str = Field(sa_column=Column("fetched", String(), FetchedValue()))

    fetched_kw: str = Field(sa_column=Column("fetched_kw", String(), server_default=FetchedValue()))

    inserted: str = Field(sa_column=Column("inserted", String(), insert_default="x"))


GeneratedDefaults()
# MYPY: error: Missing named argument "required" for "__init__" of "GeneratedDefaults"  [call-arg]
GeneratedDefaults(required="x")


class AutoIncrementPk(SQLModel, table=True):
    # `autoincrement=True` on a primary key renders as SERIAL/IDENTITY: value comes from the DB.
    auto_pk: int = Field(sa_column=Column("auto_pk", Integer, primary_key=True, autoincrement=True))


AutoIncrementPk()


class ManualPk(SQLModel, table=True):
    # A bare `primary_key=True` says nothing about who supplies the value.
    manual_pk: int = Field(sa_column=Column("manual_pk", Integer, primary_key=True))

    # `autoincrement=True` off a primary key is ignored by SQLAlchemy (plain INTEGER column).
    counter: int = Field(sa_column=Column("counter", Integer, autoincrement=True))


ManualPk()
# MYPY: error: Missing named argument "manual_pk" for "__init__" of "ManualPk"  [call-arg]
# MYPY: error: Missing named argument "counter" for "__init__" of "ManualPk"  [call-arg]
ManualPk(manual_pk=1, counter=0)

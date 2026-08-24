from __future__ import annotations

from sqlmodel import BigInteger, Column, Field, Sequence, SQLModel


class Order(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # `sqlmodel.Sequence` is `sqlalchemy.schema.Sequence` re-exported; passed positionally it
    # becomes `Column.default`, so the database assigns the value.
    number: int = Field(
        sa_column=Column(
            BigInteger,
            Sequence("example_number_seq", start=1, increment=1),
            index=True,
            nullable=False,
        )
    )

    title: str = Field()


Order()
Order(title="x")

from __future__ import annotations

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlmodel.ext.asyncio.session import AsyncSession


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


engine = create_engine("sqlite://")

with Session(engine) as session:
    result = session.execute(select(Hero))

    # When `typed_execute=true`, `execute(select(Model))` should not degrade to `Result[Any]`.
    row = result.one()
    ok_row_hero: Hero = row[0]
    bad_row_bool: bool = row[0]

    ok_hero: Hero = result.scalars().one()
    bad_bool: bool = result.scalars().one()

    result2 = session.execute(select(Hero.id))
    ok_id: int | None = result2.scalars().one()
    bad_id: int = result2.scalars().one()


async def async_execute(session: AsyncSession) -> None:
    result3 = await session.execute(select(Hero))
    _ok_hero2: Hero = result3.scalars().one()
    _bad_bool2: bool = result3.scalars().one()


async def async_execute_columns(session: AsyncSession) -> None:
    result = await session.execute(select(Hero.id, Hero.name))
    for hero_id, hero_name in result:
        _ok_id: int | None = hero_id
        _ok_name: str = hero_name
        _bad_id: int = hero_id
        _bad_name: int = hero_name

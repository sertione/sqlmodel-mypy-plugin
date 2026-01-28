from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine.cursor import CursorResult
from sqlalchemy.engine.result import Result
from sqlmodel import Field, Session, SQLModel, create_engine, delete
from sqlmodel.ext.asyncio.session import AsyncSession


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


engine = create_engine("sqlite://")

with Session(engine) as session:
    # accept SQLAlchemy (non-SQLModel) Executable statements, e.g. text(...)
    result = session.exec(text("SELECT 1"))
    ok_result: Result[Any] = result

    # Keep update/delete statements working (select correct overload).
    delete_result = session.exec(delete(Hero))
    ok_delete_result: CursorResult[Any] = delete_result


async def async_exec(session: AsyncSession) -> None:
    result2 = await session.exec(text("SELECT 1"))
    _ok_result2: Result[Any] = result2

    delete_result2 = await session.exec(delete(Hero))
    _ok_delete_result2: CursorResult[Any] = delete_result2

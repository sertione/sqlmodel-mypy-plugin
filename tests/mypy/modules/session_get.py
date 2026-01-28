from __future__ import annotations

from sqlmodel import Field, Session, SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


# Regression: `Session.get` / `AsyncSession.get` should accept SQLModel table models.
# (In downstream logs, this sometimes surfaced as an incompatible `SQLModelMetaclass` argument.)
class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field()


def sync_get(session: Session) -> None:
    user = session.get(User, 1)
    _ok_user: User | None = user


async def async_get(session: AsyncSession) -> None:
    user2 = await session.get(User, 1)
    _ok_user2: User | None = user2

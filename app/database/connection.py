"""
Async database engine, session factory, and a simple startup routine
that enables pgvector and creates tables.

For the MVP we use `Base.metadata.create_all` instead of Alembic
migrations to keep local setup simple. Once the schema stabilizes,
switch to Alembic (`alembic init`) for versioned migrations before
production launch — create_all() cannot handle schema changes safely.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings
from app.database.models import Base

settings = get_settings()

# Neon's pooler silently drops idle connections, which surfaced as
# "connection was closed in the middle of operation" 500s on /chat/test.
# pool_recycle retires pooled connections before Neon does.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """
    Enable the pgvector extension and create all tables that don't
    already exist. Safe to call every startup — it's a no-op if the
    schema is already in place.
    """
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

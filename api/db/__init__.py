
import asyncio
import os
import sys
import logging
import asyncpg
import psycopg2
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.future import select
from api.core.config import settings

engine = create_async_engine(str(settings.ASYNC_DATABASE_URI), echo=True)

async def create_extension() -> None:
    conn: asyncpg.Connection = await asyncpg.connect(
        user=settings.DB_USER,
        password=settings.DB_PASS,
        database=settings.DB_NAME,
        host=settings.DB_HOST,
    )
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXIST vector")
        logging.info("pgvector extension created or already exists")
    except Exception as ex:
        logging.error(f"Error creating pgvector extension: {ex}")
    finally:
        await conn.close()

def create_database(database_name: str, user: str, password: str, host: str, port: str) -> None:
    pass

def create_superuser() -> None:
    pass

async def init_db() -> None:
    pass
"""Alembic environment: runs migrations against DATABASE_URL."""

from logging.config import fileConfig

from alembic import context

from app.db import get_engine, metadata

if context.config.config_file_name is not None:
    fileConfig(context.config.config_file_name)

with get_engine().connect() as connection:
    context.configure(connection=connection, target_metadata=metadata)
    with context.begin_transaction():
        context.run_migrations()

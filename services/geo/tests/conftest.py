"""Database tests need a disposable PostGIS database in TEST_DATABASE_URL; they skip without it.

Local:  docker compose up -d postgis, create the database once
        (docker compose exec postgis createdb -U cropwatch cropwatch_test), then
        TEST_DATABASE_URL=postgresql+psycopg://cropwatch:change_me_locally@localhost:5432/cropwatch_test
The tables are dropped and recreated with Alembic at the start of the test session.
"""

import os
from pathlib import Path

import pytest

TEST_DB = os.environ.get("TEST_DATABASE_URL")
HERE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def client():
    if not TEST_DB:
        pytest.skip("TEST_DATABASE_URL not set")
    os.environ["DATABASE_URL"] = TEST_DB
    from alembic import command
    from alembic.config import Config
    from fastapi.testclient import TestClient

    from app.db import get_engine
    from app.main import create_app

    get_engine.cache_clear()
    cfg = Config(str(HERE / "alembic.ini"))
    cfg.set_main_option("script_location", str(HERE / "migrations"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    with TestClient(create_app()) as c:
        yield c

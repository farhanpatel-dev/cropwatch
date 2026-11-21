"""Database connection and table definitions (schema from docs/api.md, created by Alembic)."""

import os
from functools import lru_cache

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Index,
    MetaData,
    SmallInteger,
    Table,
    Text,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, REAL, UUID

metadata = MetaData()

analyses = Table(
    "analyses",
    metadata,
    Column("id", UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")),
    Column("name", Text),
    Column("source_filename", Text),
    Column("model_name", Text, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("status", Text, nullable=False, server_default="completed"),
    Column("image_bounds", Geometry("POLYGON", srid=4326, spatial_index=False)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column(
        "analysis_id",
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("class_id", SmallInteger, nullable=False),
    Column("class_name", Text, nullable=False),
    Column("confidence", REAL),
    Column("area_m2", DOUBLE_PRECISION),
    Column("geom", Geometry("MULTIPOLYGON", srid=4326, spatial_index=False), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("predictions_geom_gix", "geom", postgresql_using="gist"),
    Index("predictions_analysis_idx", "analysis_id"),
    Index("predictions_class_idx", "class_id"),
)


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example)")
    return url


@lru_cache
def get_engine() -> Engine:
    return create_engine(database_url(), pool_pre_ping=True)

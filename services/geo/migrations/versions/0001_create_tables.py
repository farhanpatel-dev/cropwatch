"""create analyses and predictions tables (schema in docs/api.md)

Revision ID: 0001
Revises:
Create Date: 2026-11-20
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute(
        """
        CREATE TABLE analyses (
          id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          name            TEXT,
          source_filename TEXT,
          model_name      TEXT NOT NULL,
          model_version   TEXT NOT NULL,
          status          TEXT NOT NULL DEFAULT 'completed',
          image_bounds    geometry(Polygon, 4326),
          created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE predictions (
          id           BIGSERIAL PRIMARY KEY,
          analysis_id  UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
          class_id     SMALLINT NOT NULL,
          class_name   TEXT NOT NULL,
          confidence   REAL,
          area_m2      DOUBLE PRECISION,
          geom         geometry(MultiPolygon, 4326) NOT NULL,
          created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX predictions_geom_gix ON predictions USING GIST (geom)")
    op.execute("CREATE INDEX predictions_analysis_idx ON predictions (analysis_id)")
    op.execute("CREATE INDEX predictions_class_idx ON predictions (class_id)")


def downgrade() -> None:
    op.execute("DROP TABLE predictions")
    op.execute("DROP TABLE analyses")

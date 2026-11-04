-- Runs once, the first time the PostGIS container creates its database.
-- Tables are created by Alembic migrations in services/geo/migrations, not here.
CREATE EXTENSION IF NOT EXISTS postgis;

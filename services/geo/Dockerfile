# Geo service image: FastAPI + SQLAlchemy/GeoAlchemy2. Build from the repo root:
#   docker compose build geo
# On start it applies Alembic migrations (creates the tables), then serves on port 8002.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# AWS Lambda Web Adapter (ADR 0004): lets this unchanged web server run on Lambda.
# It is a Lambda "extension" and does nothing when the image runs in Docker Compose.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
# Readiness by TCP: /health also checks the database, which may be waking up.
ENV AWS_LWA_PORT=8002 AWS_LWA_READINESS_CHECK_PROTOCOL=tcp

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY alembic.ini .
COPY migrations/ migrations/
COPY app/ app/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8002"]

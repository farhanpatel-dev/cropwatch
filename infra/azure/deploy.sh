#!/usr/bin/env bash
# Deploy CropWatch to Azure (ADR 0004): Container Apps for the three services, PostgreSQL
# Flexible Server (B1MS) with PostGIS, Static Web Apps (Free) for the frontend.
# Images come from GitHub Container Registry (public packages), built by build-images.sh.
#
# Usage (from the repo root, after `az login`):
#   export CW_SUFFIX=<short unique text, e.g. your initials + 3 digits>
#   export CW_DB_PASSWORD='<a strong password; never commit it>'
#   export CW_IMAGE_PREFIX=ghcr.io/<github user>/cropwatch    CW_TAG=<image tag>
#   bash infra/azure/deploy.sh
# Safe to run again: existing resources are updated, not duplicated.
set -euo pipefail

: "${CW_SUFFIX:?set CW_SUFFIX}" "${CW_DB_PASSWORD:?set CW_DB_PASSWORD}"
: "${CW_IMAGE_PREFIX:?set CW_IMAGE_PREFIX}" "${CW_TAG:?set CW_TAG}"
LOCATION=${CW_LOCATION:-canadacentral}
RG=cropwatch-rg
DB_SERVER=cropwatch-db-$CW_SUFFIX
DB_USER=cropwatch
DB_NAME=cropwatch
ENV_NAME=cropwatch-env
SWA_NAME=cropwatch-web-$CW_SUFFIX

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.DBforPostgreSQL --wait

echo "== resource group"
az group create -n "$RG" -l "$LOCATION" -o none

echo "== PostgreSQL (B1MS, free for 12 months on student/free accounts)"
if ! az postgres flexible-server show -g "$RG" -n "$DB_SERVER" -o none 2>/dev/null; then
  az postgres flexible-server create -g "$RG" -n "$DB_SERVER" -l "$LOCATION" \
    --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
    --admin-user "$DB_USER" --admin-password "$CW_DB_PASSWORD" \
    --public-access 0.0.0.0 --high-availability Disabled --backup-retention 7 -o none
fi
# Azure only allows extensions on this list; geo's first migration runs CREATE EXTENSION postgis.
az postgres flexible-server parameter set -g "$RG" -s "$DB_SERVER" \
  --name azure.extensions --value POSTGIS -o none
az postgres flexible-server db create -g "$RG" -s "$DB_SERVER" -d "$DB_NAME" -o none 2>/dev/null || true
DB_HOST=$(az postgres flexible-server show -g "$RG" -n "$DB_SERVER" --query fullyQualifiedDomainName -o tsv)
DATABASE_URL="postgresql+psycopg://$DB_USER:$CW_DB_PASSWORD@$DB_HOST:5432/$DB_NAME?sslmode=require"

echo "== Container Apps environment"
az containerapp env show -g "$RG" -n "$ENV_NAME" -o none 2>/dev/null ||
  az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOCATION" -o none

# app NAME IMAGE PORT INGRESS CPU MEMORY [extra args...]
app() {
  local name=$1 image=$2 port=$3 ingress=$4 cpu=$5 mem=$6
  shift 6
  if az containerapp show -g "$RG" -n "$name" -o none 2>/dev/null; then
    az containerapp update -g "$RG" -n "$name" --image "$image" --cpu "$cpu" --memory "$mem" "$@" -o none
  else
    az containerapp create -g "$RG" -n "$name" --environment "$ENV_NAME" --image "$image" \
      --target-port "$port" --ingress "$ingress" --cpu "$cpu" --memory "$mem" \
      --min-replicas 0 --max-replicas 1 "$@" -o none
  fi
}

echo "== inference (internal; 2 vCPU / 4 GiB; sleeps when idle)"
app cropwatch-inference "$CW_IMAGE_PREFIX-inference:$CW_TAG" 8001 internal 2.0 4.0Gi

echo "== geo (internal)"
az containerapp show -g "$RG" -n cropwatch-geo -o none 2>/dev/null ||
  app cropwatch-geo "$CW_IMAGE_PREFIX-geo:$CW_TAG" 8002 internal 0.25 0.5Gi \
    --secrets "database-url=$DATABASE_URL" --env-vars "DATABASE_URL=secretref:database-url"
az containerapp secret set -g "$RG" -n cropwatch-geo --secrets "database-url=$DATABASE_URL" -o none
app cropwatch-geo "$CW_IMAGE_PREFIX-geo:$CW_TAG" 8002 internal 0.25 0.5Gi

echo "== frontend site (Static Web Apps, Free)"
az staticwebapp show -g "$RG" -n "$SWA_NAME" -o none 2>/dev/null ||
  az staticwebapp create -g "$RG" -n "$SWA_NAME" -l eastus2 --sku Free -o none
WEB_URL="https://$(az staticwebapp show -g "$RG" -n "$SWA_NAME" --query defaultHostname -o tsv)"

echo "== gateway (public)"
# Apps in one Container Apps environment reach each other by app name over HTTP.
app cropwatch-gateway "$CW_IMAGE_PREFIX-gateway:$CW_TAG" 8000 external 0.5 1.0Gi \
  --env-vars INFERENCE_URL=http://cropwatch-inference GEO_URL=http://cropwatch-geo \
  MAX_UPLOAD_MB=50 "CORS_ORIGINS=$WEB_URL,http://localhost:5173"
API_URL="https://$(az containerapp show -g "$RG" -n cropwatch-gateway --query properties.configuration.ingress.fqdn -o tsv)"

echo "== build and upload the frontend"
(cd frontend && npm ci --silent && VITE_API_URL="$API_URL" npm run build --silent)
TOKEN=$(az staticwebapp secrets list -g "$RG" -n "$SWA_NAME" --query properties.apiKey -o tsv)
npx --yes @azure/static-web-apps-cli@2 deploy frontend/dist --deployment-token "$TOKEN" --env production

echo
echo "Website: $WEB_URL"
echo "API:     $API_URL  (docs at $API_URL/docs)"

#!/usr/bin/env bash
# Build the three service images for Azure (linux/amd64; Macs build arm64 by default) and push
# them to GitHub Container Registry. Needs `docker login ghcr.io` first.
# Usage: CW_IMAGE_PREFIX=ghcr.io/<github user>/cropwatch CW_TAG=<tag> bash infra/azure/build-images.sh
set -euo pipefail
: "${CW_IMAGE_PREFIX:?set CW_IMAGE_PREFIX}" "${CW_TAG:?set CW_TAG}"
for svc in inference geo gateway; do
  echo "== $svc"
  docker buildx build --platform linux/amd64 -t "$CW_IMAGE_PREFIX-$svc:$CW_TAG" --push "services/$svc"
done

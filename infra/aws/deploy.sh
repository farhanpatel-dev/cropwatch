#!/usr/bin/env bash
# Deploy CropWatch to AWS (ADR 0004). Run from the repo root after `aws login` (or
# `aws configure`). Safe to run again: it updates the existing stack.
#
#   export CW_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require'  # Neon
#   bash infra/aws/deploy.sh
#
# Builds the three service images for arm64 (native on Apple silicon), pushes them to ECR,
# runs the database migrations, deploys infra/aws/template.yaml, then uploads the website.
set -euo pipefail

: "${CW_DATABASE_URL:?set CW_DATABASE_URL to the Neon connection string (postgresql+psycopg://...)}"
REGION=${AWS_REGION:-${CW_REGION:-ca-central-1}}
STACK=${CW_STACK:-cropwatch}
TAG=$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- services || echo "-dirty")
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REGISTRY="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"
export AWS_REGION=$REGION

echo "== images ($TAG) -> $REGISTRY"
aws ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
for svc in inference geo gateway; do
  repo="cropwatch-$svc"
  if ! aws ecr describe-repositories --repository-names "$repo" >/dev/null 2>&1; then
    aws ecr create-repository --repository-name "$repo" >/dev/null
    # Keep only the 2 newest images: ECR storage is billed per GB.
    aws ecr put-lifecycle-policy --repository-name "$repo" --lifecycle-policy-text \
      '{"rules":[{"rulePriority":1,"selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":2},"action":{"type":"expire"}}]}' >/dev/null
  fi
  # Lambda needs a single-platform image without attestations.
  docker buildx build --platform linux/arm64 --provenance=false --sbom=false \
    -t "$REGISTRY/$repo:$TAG" --load "services/$svc"
  docker push -q "$REGISTRY/$repo:$TAG"
done

echo "== database migrations (Neon)"
docker run --rm -e DATABASE_URL="$CW_DATABASE_URL" "$REGISTRY/cropwatch-geo:$TAG" alembic upgrade head

echo "== stack $STACK in $REGION"
aws cloudformation deploy --stack-name "$STACK" --template-file infra/aws/template.yaml \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset --parameter-overrides \
  "InferenceImageUri=$REGISTRY/cropwatch-inference:$TAG" \
  "GeoImageUri=$REGISTRY/cropwatch-geo:$TAG" \
  "GatewayImageUri=$REGISTRY/cropwatch-gateway:$TAG" \
  "DatabaseUrl=$CW_DATABASE_URL" \
  ${CW_INFERENCE_MEMORY_MB:+"InferenceMemoryMb=$CW_INFERENCE_MEMORY_MB"}

output() {
  aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
SITE_URL=$(output SiteUrl)
API_URL=$(output ApiUrl)

echo "== website"
(cd frontend && npm ci --silent && VITE_API_URL="${API_URL%/}" npm run build --silent)
BUCKET=$(output SiteBucket)
aws s3 sync frontend/dist "s3://$BUCKET" --delete --exclude index.html \
  --cache-control "public,max-age=31536000,immutable" >/dev/null
aws s3 cp frontend/dist/index.html "s3://$BUCKET/index.html" --cache-control "no-cache" >/dev/null
aws cloudfront create-invalidation --distribution-id "$(output DistributionId)" \
  --paths /index.html >/dev/null

echo
echo "Website: $SITE_URL"
echo "API:     $API_URL  (docs at ${API_URL}docs)"

# ADR 0004: Host the whole stack on AWS always-free services

- Status: Accepted
- Date: 2026-11-21
- Owner: Ibrahim (infra)

## Context
The plan (docs/architecture.md, "Hosting decision") was: frontend on Vercel, backend as a recorded demo, Azure only as a short lab. The goal is now a live site that works from any device, on AWS (useful experience for the team). Inference needs about 1.2 GB of RAM (ADR 0003); geo needs PostgreSQL with PostGIS.

AWS for students in November 2026: AWS Educate gives no credits since 2023; Student Rewards gives 30 USD after badges; new accounts get 100-200 USD credits on a Free plan that ends after 6 months, and a card is required. The old 12-month Free Tier ended for new accounts on 2025-07-15. What stays free are the "always free" offers, among them Lambda (1 million requests and 400,000 GB-seconds per month) and CloudFront (1 TB per month). Checked 2026-11-21; re-check the AWS Free Tier page before relying on it.

## Decision
| Part | Service | Notes |
| --- | --- | --- |
| frontend | S3 (private) + CloudFront | built with `VITE_API_URL` = gateway URL |
| gateway | Lambda (container image, arm64), public function URL | 512 MB, 100 s timeout |
| inference | Lambda (container image, arm64), function URL with IAM auth | 10,240 MB (6 vCPU), 90 s |
| geo | Lambda (container image, arm64), function URL with IAM auth | 512 MB; migrations run by deploy.sh |
| uploads | S3 bucket, objects deleted after 1 day | browser uploads with a presigned POST |
| database | Neon free plan (PostgreSQL with PostGIS), outside AWS | no always-free PostgreSQL on AWS |
| images | ECR, 2 newest images kept per repository | |

The same Docker images run in Compose and on Lambda: the AWS Lambda Web Adapter (an extension copied into each image) forwards Lambda events to the unchanged FastAPI server. Only the gateway is public; inference and geo accept only requests signed with the gateway's IAM role (SigV4), so the HTTP-only rule between services still holds.

Lambda limits a request body to 6 MB, but uploads may be 50 MB. So on AWS: the browser asks the gateway for an upload slot (`POST /api/v1/uploads`), posts the file to S3, then calls `POST /api/v1/analyses` with `upload_key`; the gateway passes inference a short-lived presigned S3 URL (`image_url`). Inference only fetches URLs under `IMAGE_URL_PREFIXES` (the upload bucket). Locally nothing changes: `/api/v1/uploads` answers 501 and the browser sends the file directly.

Everything is in `infra/aws/template.yaml` (CloudFormation) and `infra/aws/deploy.sh`.

## Alternatives considered
- **Azure for Students** (no card, 12 months of PostgreSQL, Container Apps free grant). The simplest free option; not chosen because the team wants AWS experience.
- **One EC2 server running Docker Compose on credits.** Least new code, but stops being free after the 6-month Free plan.
- **ECS Fargate or App Runner.** No always-free allowance.
- **DynamoDB (always free) instead of PostGIS.** No spatial queries; geo would be rewritten.

## Consequences
- Good: a public HTTPS site that sleeps when unused. Within the always-free Lambda and CloudFront allowances; a 4000 px analysis uses about 300 GB-seconds at 10 GB (estimate), so roughly 1,300 large or tens of thousands of small analyses per month.
- Not quite zero: ECR storage (about 0.10 USD per GB-month) and S3 requests cost cents per month, paid from credits first. Neon's free plan has its own limits (storage, compute hours).
- A card is required to open the AWS account. Set a budget alert (AWS Budgets) on day one.
- Cold start: after a quiet period the first request starts the containers (estimate: several seconds, plus Neon waking up). The frontend calls `GET /api/v1/model-info` on page load to wake inference early.
- Speed: 10,240 MB gives 6 vCPU, about the CPU of the M2 measurement in ADR 0003; re-measure 4000 px on Lambda. New AWS accounts may be capped at 3,008 MB per function until the quota is raised; then lower the hosted image limit.
- The Agriculture-Vision terms are non-commercial; a free public demo fits them.

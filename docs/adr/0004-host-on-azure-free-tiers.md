# ADR 0004: Host the whole stack on Azure free tiers

- Status: Accepted
- Date: 2026-11-20
- Owner: Ibrahim (infra)

## Context
The plan (docs/architecture.md, "Hosting decision") was: frontend on Vercel, backend as a recorded demo of the local stack, Azure only as a short lab deleted afterwards. The goal is now a live site that works from any device. Inference needs about 1.2 GB of RAM (ADR 0003), which free hosts with 512 MB cannot run, and the geo service needs PostgreSQL with PostGIS.

Ibrahim has Azure for Students (Western University): 100 USD credit, no card, spending stops when the credit is used up. It includes 12 months of PostgreSQL Flexible Server B1MS (750 h/month, 32 GB) and the always-free Container Apps grant (180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month). Checked 2026-11-20; re-check the offer pages before relying on them.

## Decision
| Part | Azure service | Size |
| --- | --- | --- |
| frontend | Static Web Apps, Free plan | static files from `npm run build` |
| gateway | Container Apps (Consumption), external ingress | 0.5 vCPU, 1 GiB, 0-1 replicas |
| inference | Container Apps, internal ingress | 2 vCPU, 4 GiB, 0-1 replicas |
| geo | Container Apps, internal ingress | 0.25 vCPU, 0.5 GiB, 0-1 replicas |
| database | PostgreSQL Flexible Server B1MS, PostGIS allow-listed | 32 GB |
| images | GitHub Container Registry (public packages) | linux/amd64 |

Apps scale to zero when idle, so they only use the free grant while someone is using the site. Scripts: `infra/azure/build-images.sh` and `infra/azure/deploy.sh`.

## Alternatives considered
- **Vercel + recorded demo (the old default).** Free, but the backend is not live.
- **One Azure VM running Docker Compose.** Simplest, but the free-tier VM (B1s, 1 GiB) is too small for inference, and larger VMs use the credit every hour, also when nobody visits.
- **Azure Container Registry.** Paid from the first day (Basic tier); GHCR is free for public images.
- **Oracle Cloud Always Free VM.** Enough RAM, but needs a card and is not the team's cloud.

## Consequences
- Good: a public URL that works on any device, at no cost while the grant and the 12 free database months last.
- Cold start: after a quiet period the first request wakes the apps (estimate: tens of seconds; measure after deploying). The frontend calls `GET /api/v1/model-info` on page load to wake inference early.
- Speed: inference has 2 vCPU instead of the 6-16 CPUs measured in ADR 0003, so a 4000 px image may exceed the 60 s gateway timeout. Measure on Azure and lower the limit for the hosted site if needed.
- Cost risk: the database is free for 12 months only. Before then, move it (for example to a free Neon project) or delete the resource group. Log Analytics (created with the Container Apps environment) is free only up to its monthly ingestion allowance.
- The images and the model are public on GHCR. The model's terms are non-commercial (Agriculture-Vision), which matches a public demo.
- Secrets: the database password lives only in a Container Apps secret and in the person's shell when deploying; never in Git.

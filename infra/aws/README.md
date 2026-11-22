# AWS hosting (ADR 0004)

One-time setup:
1. AWS account (card needed), then in the console: Billing -> Budgets -> create a 1 USD budget alert.
2. `brew install awscli`, then `aws login` (or `aws configure` with an access key). Region: us-east-2 (same region as the Neon database; new free-plan accounts are locked to the region chosen at sign-up).
3. Neon (https://neon.tech, free plan): create a project, run `CREATE EXTENSION postgis;` in its SQL
   editor, and copy the connection string. Change `postgresql://` to `postgresql+psycopg://`.

Deploy (from the repo root; Docker must be running):
```bash
# put the Neon string in .env (ignored by Git) as one line:
#   CW_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/neondb?sslmode=require
bash infra/aws/deploy.sh     # prints the website and API URLs
```
Run it again after any change. Remove everything: `aws cloudformation delete-stack --stack-name cropwatch`
(empty the two S3 buckets first), then delete the three `cropwatch-*` ECR repositories.

Automatic deploys from GitHub (`.github/workflows/deploy.yml`): every push to main that touches
`services/`, `frontend/` or `infra/aws/` runs `deploy.sh` on a GitHub arm64 runner. You can also
start it by hand: Actions -> Deploy -> Run workflow. One-time setup, from the repo root after `aws login`:
```bash
# 1. the Neon string, encrypted in SSM Parameter Store (free), so GitHub never sees it
aws ssm put-parameter --name /cropwatch/database-url --type SecureString \
  --value "$(grep -E '^CW_DATABASE_URL=' .env | cut -d= -f2-)"
# 2. the role GitHub uses (OIDC, no stored keys; only main of farhanpatel-dev/cropwatch may use it)
aws cloudformation deploy --stack-name cropwatch-github --template-file infra/aws/github-deploy.yaml \
  --capabilities CAPABILITY_NAMED_IAM
```

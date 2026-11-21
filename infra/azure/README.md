# Azure hosting (ADR 0004)

One-time setup (macOS: `brew install azure-cli`):
1. Azure for Students with your school email: https://azure.microsoft.com/free/students
2. `az login`
3. GitHub token that can push packages: `gh auth refresh -h github.com -s write:packages`, then
   `gh auth token | docker login ghcr.io -u <github user> --password-stdin`

Deploy:
```bash
export CW_IMAGE_PREFIX=ghcr.io/<github user>/cropwatch CW_TAG=$(git rev-parse --short HEAD)
bash infra/azure/build-images.sh        # 3 images for linux/amd64 (slow on Apple silicon: emulated)
# On GitHub: Packages -> each cropwatch-* package -> Package settings -> Change visibility -> Public
export CW_SUFFIX=<unique, e.g. sia123> CW_DB_PASSWORD='<strong password>'
bash infra/azure/deploy.sh              # prints the website and API URLs
```
Update later: build with a new `CW_TAG`, run `deploy.sh` again.

Stop everything (deletes all data): `az group delete -n cropwatch-rg`.

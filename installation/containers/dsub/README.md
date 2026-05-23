# dsub on AoU Researcher Workbench

GPU batch jobs against the unified container image at `installation/containers/`.
dsub ≈ SLURM's `sbatch`. All `dsub`/`gcloud` commands run **inside a Workbench
Jupyter terminal** — AoU's VPC perimeter blocks them from a laptop.

## How AoU handles custom container images

**Cloud Build is not enabled on AoU workspace projects.** Researchers can't run
`wb gcloud builds submit` against their workspace — the Service Usage API is
disabled and the Cloud Build service account isn't provisioned. AoU's documented
Docker workflow routes around this with a **central GAR remote-repository proxy**
that fronts Docker Hub:

```
$ARTIFACT_REGISTRY_DOCKER_REPO
  = us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod
```

This env var is set in every Workbench environment. The Batch VM pulls from this
proxy (it's inside the AoU perimeter); it can't reach Docker Hub directly.

**Workflow:**

1. Build the image **outside AoU** (laptop, GitHub Actions, etc.).
2. Push it as a **public** image to Docker Hub.
3. Reference it via the proxy in dsub:
   `--image $ARTIFACT_REGISTRY_DOCKER_REPO/<dockerhub-user>/<image>:<tag>`

Private images aren't self-service — you'd have to email
`support@researchallofus.org`.

**Sources:**
- [AoU — Using Docker Images on the Workbench](https://support.researchallofus.org/hc/en-us/articles/21179878475028)
- [AoU — Use dsub in the Researcher Workbench](https://support.researchallofus.org/hc/en-us/articles/4692986669332)

## Build the image (outside AoU)

### Option A — Docker locally (Mac/Linux laptop)

```bash
cd ~/Delphi
docker login                                    # one-time
SHA=$(git rev-parse --short HEAD)
docker build -t <dockerhub-user>/delphi:$SHA installation/containers/
docker push <dockerhub-user>/delphi:$SHA
```

### Option B — GitHub Actions (no local Docker needed)

Add `.github/workflows/build-image.yml` to the Delphi repo:

```yaml
name: build-image
on:
  push:
    branches: [main]
    paths: ['installation/containers/**']
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: installation/containers/
          push: true
          tags: |
            <dockerhub-user>/delphi:${{ github.sha }}
            <dockerhub-user>/delphi:latest
```

Add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` to the repo's GitHub Actions
secrets.

Tag with git SHA, not `latest`, for reproducibility and cache friendliness
(see Startup time below).

## Startup time via the proxy

The AoU proxy lives in `us-central1` (same region as Batch VMs) and caches
per tag:

| Scenario | Path | Time |
|---|---|---|
| First job ever with `image:tag` | Docker Hub → proxy → VM | ~30–60 s |
| Repeated jobs, same tag | proxy → VM (in-region) | ~10–30 s |

The cold-pull penalty is paid **once per new tag**, not per job. Reuse SHAs
across reruns to keep pulls fast.

## One-time workspace setup (job data, not builds)

A Workbench-managed bucket as scratch space for dsub inputs / outputs / logs:

```bash
wb resource create gcs-bucket --id=ws_files \
  --description="Scratch bucket for dsub job I/O."
```

Referenced as `${WORKBENCH_ws_files}` in dsub flags (below). `wb resource list`
to verify.

## GitHub PAT for cloning the private repo

The entrypoint clones the Delphi repo at runtime via
`git clone $DELPHI_REPO`. Since the repo is private, the Batch VM needs
auth. The entrypoint handles this transparently: if `$GH_TOKEN` is set
in the container env, the clone URL is rewritten to
`https://x-access-token:${GH_TOKEN}@github.com/...`. Otherwise the clone
runs unchanged (so public-repo testing on your laptop still works).

One-time setup:

1. On GitHub: Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → **Generate new token**. Scope to the Delphi
   repo only; Repository permissions → Contents = Read; set a 90-day
   expiry. Copy the token (`github_pat_...`).
2. On the Jupyter VM: store the token in a mode-600 file.
   ```bash
   nano ~/.gh_token        # paste, save
   chmod 600 ~/.gh_token
   ```
3. Pass via dsub `--env GH_TOKEN=$(cat ~/.gh_token)` (see below).

When the token expires, regenerate at the same GitHub URL and overwrite
`~/.gh_token`. No image rebuild needed.

## Submit a job

`train.sh`:
```bash
#!/bin/bash
set -e
# The entrypoint has already cloned the repo and `pip install -e .`'d it.
python -m delphi.train --input "$TRAIN_DATA" --out "$OUT_DIR"
```

Submission:
```bash
SHA=<the git sha you built>
IMAGE=$ARTIFACT_REGISTRY_DOCKER_REPO/<dockerhub-user>/delphi:$SHA

dsub --provider google-batch \
  --project $GOOGLE_CLOUD_PROJECT \
  --regions us-central1 \
  --service-account $GOOGLE_SERVICE_ACCOUNT_EMAIL \
  --network global/networks/network \
  --subnetwork regions/us-central1/subnetworks/subnetwork \
  --use-private-address \
  --image $IMAGE \
  --machine-type n1-standard-8 \
  --accelerator-type nvidia-tesla-t4 \
  --accelerator-count 1 \
  --env DELPHI_BRANCH=main \
  --env GH_TOKEN=$(cat ~/.gh_token) \
  --input TRAIN_DATA=${WORKBENCH_ws_files}/data/train.parquet \
  --output-recursive OUT_DIR=${WORKBENCH_ws_files}/runs/run1/ \
  --logging ${WORKBENCH_ws_files}/runs/run1/logs/ \
  --script train.sh
```

The first six flags are **mandatory plumbing for AoU**; the rest control
the actual job. Without `--service-account` / `--network` / `--subnetwork`
/ `--use-private-address` you'll hit a sequence of IAM and VPC-SC errors.

Flag cheat sheet:

| Flag | Purpose |
|---|---|
| `--provider google-batch` | which compute backend to use (Google Cloud Batch) |
| `--project` | GCP project for billing / quota |
| `--regions` | region where the VM runs (must be `us-central1` on AoU) |
| `--service-account` | identity the Batch VM authenticates as; AoU requires your pet SA |
| `--network` / `--subnetwork` | AoU's VPC (literally `network` / `subnetwork`); full path form required |
| `--use-private-address` | no external IP; required by AoU's VPC-SC perimeter |
| `--image` | container image, must go through the AoU proxy |
| `--machine-type` | VM shape (n1-* required for T4/P100/V100) |
| `--accelerator-type/-count` | GPU spec |
| `--env KEY=VAL` | env vars inside the container (e.g. `DELPHI_BRANCH`) |
| `--env GH_TOKEN=...` | runtime GitHub auth for cloning the private repo |
| `--input KEY=gs://...` | localise one file; `$KEY` inside container is the local path |
| `--input-recursive KEY=gs://.../` | localise a whole directory |
| `--output-recursive KEY=gs://.../` | on exit 0, upload local `$KEY` directory back to GCS |
| `--logging gs://.../` | where stdout/stderr go (GCS dir) |
| `--script` | the script to run; dsub uploads it from your local path |
| `--wait` | block locally until job reaches a terminal state |

## What the flags actually mean

If something breaks, knowing the underlying concepts saves time. Each flag
corresponds to a real GCP / AoU concept:

**`--provider`** — dsub is a *frontend*. The provider is the backend that
actually runs the job. `google-batch` = GCP's managed batch service (≈
SLURM scheduler + autoscaling node pool). `local` runs on your current
machine; `google-cls-v2` is the deprecated Cloud Life Sciences API that
AoU migrated off in 2025.

**`--project`** — A GCP project is the billing + IAM boundary. Costs are
charged here; VMs live in this project's quota. `$GOOGLE_CLOUD_PROJECT`
is set by Workbench to your workspace's underlying project.

**`--service-account`** — A non-human Google identity (looks like
`pet-xxx@<project>.iam.gserviceaccount.com`). The Batch VM authenticates as
this SA when calling other GCP APIs (read input from GCS, pull the image,
write outputs). Without this flag, Batch tries to run the VM as the
project's default Compute Engine SA — which on AoU your pet SA isn't
allowed to impersonate, so the job fails with `caller does not have
permission to act as service account`. Pointing at the pet SA via
`$GOOGLE_SERVICE_ACCOUNT_EMAIL` gives the VM the same permissions you have.
Mental model: the SA is the badge the VM wears; the badge decides which
doors it can open.

**`--network` / `--subnetwork`** — A **VPC** is a GCP project's private
network. Every VM is attached to one. The VPC has *networks* (top-level)
and *subnetworks* (one per region, each with its own IP range). AoU
provisions a custom VPC, single network literally called `network` and
subnetwork `subnetwork` (visible in `gcloud projects describe` labels).
Batch's API requires the full-path form
(`global/networks/<name>`, `regions/<region>/subnetworks/<name>`); short
names get rejected with `network is not matching the expected format`.
On non-AoU GCP you'd usually get the default VPC for free; AoU's custom
one isolates workspace traffic at the network layer.

**`--use-private-address`** — Every GCP VM normally has *two* IP
addresses: an **external** (public, internet-routable) and an **internal**
(private, VPC-only). This flag tells Batch: internal IP only. Why AoU
mandates it: VPC Service Controls assumes no VM inside the perimeter can
reach the public internet directly — an external IP would punch a hole.
AoU rejects any Batch job that requests one (`external ip address must be
disabled`). The VM can still pull container images because
`$ARTIFACT_REGISTRY_DOCKER_REPO` lives inside the perimeter and is
reachable over private Google routes (Private Service Connect).

**`--image`** — The container image to pull and run. On AoU it must go
through the proxy at `$ARTIFACT_REGISTRY_DOCKER_REPO/<...>` because the
Batch VM has no external IP.

**`--machine-type`** — VM shape (CPU + RAM in one). Like SLURM's
`--partition` + `--cpus-per-task` combined. `n1-standard-8` = 8 vCPUs
and 30 GB RAM. GPU types constrain machine families: T4/P100/V100 only
attach to n1; A100/L4 require a2/g2.

**`--accelerator-type` / `--accelerator-count`** — GPU spec. AoU has T4
quota by default; A100 needs an approval step.

**`--env KEY=VAL`** — Plain environment variables inside the container,
the same as `docker run -e KEY=VAL`. Use these to parameterize the
entrypoint (e.g. `DELPHI_BRANCH`).

**`--input KEY=gs://...`** — Before your script runs, dsub copies the GCS
object to a local path inside the container and exports `$KEY` pointing at
that local path. `--input-recursive` does the same for whole directories.
SLURM analogy: staging input files to `$TMPDIR` before `srun`.

**`--output-recursive KEY=gs://.../`** — Inverse direction: after your
script exits 0, dsub uploads the local directory at `$KEY` back to GCS.
**Only on exit 0** — non-zero exits skip delocalization, so write critical
artifacts (checkpoints) directly via `gsutil cp` from inside the script.

**`--logging gs://.../`** — Where stdout/stderr land. Streamed during the
run if you `gsutil cat`; finalised at job exit.

**`--script`** — Your job script (local path on the submitter). dsub
uploads it and runs it as the container's entrypoint command. Multi-task
fan-out uses `--tasks` (a TSV of per-task params) instead.

**`--wait`** — Block the submitter until the job reaches a terminal state
(SUCCESS / FAILURE / CANCELED). Omit for fire-and-forget; you'll get the
job ID immediately and check on it later with `dstat`.

## The mental picture

When you `dsub ...`:

1. dsub serialises your flags into a Google Batch API request.
2. Batch (authenticated as your pet SA via dsub) creates a job spec:
   "spin up a VM in project X, attached to AoU's `network`/`subnetwork`,
   internal IP only, running as the pet SA, pull image Z, run script S."
3. Batch provisions the VM (~30 s).
4. The VM authenticates as the pet SA, pulls the image from the AoU proxy
   over private routes.
5. Runs your script.
6. Uploads stdout/stderr to the `--logging` GCS path, then
   `--output-recursive` paths.
7. Tears the VM down.
8. dsub returns (immediately, or after the job ends if `--wait`).

## Monitor / debug

```bash
dstat --provider google-batch --project $GOOGLE_CLOUD_PROJECT --jobs <job-id>
dstat ... --jobs <job-id> --full        # full detail
ddel  ... --jobs <job-id>               # cancel
gsutil cat ${WORKBENCH_ws_files}/runs/run1/logs/log.txt   # tail logs mid-run
```

SLURM mapping: `dsub` ≈ `sbatch`, `dstat` ≈ `squeue`, `ddel` ≈ `scancel`.

## Network reachability from the perimeter

github.com is reachable from inside the AoU perimeter (confirmed in
testing from a Workbench Jupyter terminal). Runtime cloning works.

## Gotchas

- **No Cloud Build on AoU.** Don't try `wb gcloud builds submit` — the API
  isn't initialized on workspace projects. Build externally and pull through
  the proxy. (`$ARTIFACT_REGISTRY_DOCKER_REPO` is the supported pattern.)
- **Public Docker Hub only**: the proxy fronts public images. Private images
  need a support ticket.
- **Use `${WORKBENCH_<id>}`, not `$WORKSPACE_BUCKET`.** The AoU-style env var
  often points at a bucket that doesn't exist on Verily Workbench 2.0.
  Workbench-managed resources expose themselves as `${WORKBENCH_<resource-id>}`.
- **Use `$GOOGLE_CLOUD_PROJECT`, not `$GOOGLE_PROJECT`.** Both may be set but
  the Verily docs standardise on the former.
- **`--output*` runs only on exit 0.** If the job crashes mid-training, dsub
  won't delocalize. Write checkpoints from inside `train.sh` via
  `gsutil cp /tmp/ckpt.pt gs://...` to survive failures.
- **Tag with git SHA, not `latest`.** `latest` defeats the proxy's per-tag
  cache and makes "which build is running?" unanswerable.
- **GPU types**: T4/P100/V100 only on n1. A100/L4 require a2/g2 — verify quota
  before assuming.
- **Container runs as root.** See header comment in `../Dockerfile` for why.
- **PAT in dsub env is visible in Batch metadata.** Fine-grained PAT scoped
  to one repo, read-only contents, with an expiry minimises blast radius.
  Rotate regularly.

## References

- [AoU — Using Docker Images on the Workbench](https://support.researchallofus.org/hc/en-us/articles/21179878475028)
- [AoU — Use dsub in the Researcher Workbench](https://support.researchallofus.org/hc/en-us/articles/4692986669332)
- [AoU — Overview of Batch Processing](https://support.researchallofus.org/hc/en-us/articles/4692418691732)
- [Verily Workbench — Create container images](https://support.workbench.verily.com/docs/guides/cloud_apps/advanced_app_usage/create_container_images/) *(AoU diverges — informational only)*

## File layout

```
installation/containers/
├── Dockerfile           # shared image
├── entrypoint.sh        # clones $DELPHI_REPO @ $DELPHI_BRANCH, pip install -e .
├── requirements.txt
└── dsub/
    └── README.md        # this file
```

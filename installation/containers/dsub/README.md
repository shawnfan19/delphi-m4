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
  --image $IMAGE \
  --machine-type n1-standard-8 \
  --accelerator-type nvidia-tesla-t4 \
  --accelerator-count 1 \
  --env DELPHI_BRANCH=main \
  --input TRAIN_DATA=${WORKBENCH_ws_files}/data/train.parquet \
  --output-recursive OUT_DIR=${WORKBENCH_ws_files}/runs/run1/ \
  --logging ${WORKBENCH_ws_files}/runs/run1/logs/ \
  --script train.sh
```

Flag cheat sheet:

| Flag | Purpose |
|---|---|
| `--image` | full path including the AoU proxy prefix |
| `--machine-type` | n1-* required for T4/P100/V100 |
| `--accelerator-type/-count` | GPU spec |
| `--env KEY=VAL` | env vars inside the container (e.g. `DELPHI_BRANCH`) |
| `--input KEY=gs://...` | localise one file; `$KEY` inside container is the local path |
| `--input-recursive KEY=gs://.../` | localise a whole directory |
| `--output-recursive KEY=gs://.../` | on exit 0, upload local `$KEY` directory back to GCS |
| `--logging gs://.../` | where stdout/stderr go (GCS dir) |
| `--script` | the script to run; dsub uploads it from your local path |

## Monitor / debug

```bash
dstat --provider google-batch --project $GOOGLE_CLOUD_PROJECT --jobs <job-id>
dstat ... --jobs <job-id> --full        # full detail
ddel  ... --jobs <job-id>               # cancel
gsutil cat ${WORKBENCH_ws_files}/runs/run1/logs/log.txt   # tail logs mid-run
```

SLURM mapping: `dsub` ≈ `sbatch`, `dstat` ≈ `squeue`, `ddel` ≈ `scancel`.

## Verify GitHub reachability from the perimeter

The entrypoint does `git clone gerstung-lab/Delphi.git` at runtime. AoU's
perimeter restricts outbound traffic; github.com may or may not be reachable
from Batch VMs. **Test once from a Jupyter terminal** (same perimeter rules):

```bash
git clone https://github.com/gerstung-lab/Delphi.git /tmp/test-clone
```

- If this works, runtime cloning will work in Batch too — current setup is fine.
- If it fails, **bake the code into the image at build time** instead. Patch
  the Dockerfile to `COPY . /workspace/Delphi && pip install -e .` and remove
  the clone from `entrypoint.sh`. Trade-off: rebuild image per code change,
  but no network dependency at runtime.

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

# LAUNCHPAD.md — submitting jobs to SLURM with `submit`

How to run Delphi training/eval/plot scripts as cluster jobs. The launcher is a
tiny package (`slurm-submit`) that turns a command into an `sbatch` script and
submits it. Source lives at `/hps/software/users/birney/sfan/launchpad`
(`src/slurm_submit/cli.py` is the whole thing — no README, the code is the spec).

> AoU sibling: the dsub/GCS launcher `dsubmit` mirrors this for the All of Us
> workbench. This doc is the **SLURM** launcher (`submit`).

## The `submit` command

`submit` is a console script (`pyproject.toml` → `submit = slurm_submit.cli:main`)
installed in the `delphi-cf-torch2.3` env, so it is already on `PATH` when that
env is active. It depends only on `omegaconf` + `pyyaml`.

```
submit <script> [script-args …]  --  [key=value launcher config …]
```

- The command is split on a **standalone `--` token** (`argv.index("--")`). Flags
  like `--input_path` are *not* the separator — only a bare `--` is.
- **Everything before `--`** becomes the command to run. It is executed as
  `python <script> <script-args> "$@"`. **Do not prefix with `python`** — the
  template already adds it (`submit python foo.py …` runs `python python foo.py`).
- **Everything after `--`** is parsed by OmegaConf as launcher config (resources,
  partition, sweeps, …). If there is no `--`, all args are the command and
  defaults are used for the launcher.

### What the generated job does

The job script (written to `/dev/shm/sbatch_*.sh`, then `sbatch`-ed) runs **from the
directory where you call `submit`** and does:

```bash
#SBATCH … (time, mem, gres=gpu, job-name, output/error, …)
source ~/.bashrc
set -a; source .env; set +a          # <-- needs a .env in the CWD
module load cuda/<cuda_version>
micromamba activate $PYTHON_ENV       # <-- PYTHON_ENV comes from .env
python <script> "$@"
```

**Two prerequisites in the submission directory:**
1. A **`.env`** defining at least `PYTHON_ENV=<conda/micromamba env>`. The Delphi
   repo root already has one (`PYTHON_ENV=delphi-cf-torch2.3` plus the
   `DELPHI_CKPT_*`/`DELPHI_DATA_DIR`/`DELPHI_DATASET` vars), so submitting from the
   repo root just works.
2. A **`slurm/`** directory for logs — stdout/stderr default to
   `slurm/slurm-%j.out` / `slurm/slurm-%j.err` (relative). Create it once:
   `mkdir -p slurm`. (Missing dir ⇒ the job fails to write output.)

## Launcher config (after `--`)

OmegaConf `key=value` pairs (`RunConfig` in `cli.py`). Defaults in parentheses:

| key | default | meaning |
|---|---|---|
| `gpu` | `true` | request a GPU (`--gres=gpu:…`) |
| `gpu_num` | `1` | GPUs; **>1 wraps the command in `torchrun --standalone --nproc-per-node=N`** |
| `gpu_type` | none | `a100` or `v100` → `--gres=gpu:<type>:N` |
| `memory` | `32` | GB → `--mem=NG` |
| `time` | `3.0` | hours (fractional ok; `6.5` → `06:30:00`) |
| `cpu_num` | none | `--cpus-per-task` |
| `task_num` / `node_num` | none | `--ntasks` / `--nodes` |
| `partition` | none | `standard`, `production`, `research`, `debug`, `datamover`, `datamover_debug`. **Usually omit** — the scheduler routes GPU jobs automatically (short jobs land in `short_gpu`). |
| `job_name` | none | `--job-name` (handy for `squeue`) |
| `cuda_version` | `11.8.0` | `module load cuda/<v>` |
| `stdout` / `stderr` | `slurm/slurm-%j.{out,err}` | log paths |
| `mail_type` | none | `BEGIN`/`END`/`FAIL`/`ALL`/`TIME_LIMIT_90`/`TIME_LIMIT_80`/`TIME_LIMIT_50`/`ARRAY_TASKS` |
| `dry` | `false` | **preview**: write+print the sbatch script, do not submit |
| `config` | none | load a YAML of these keys (merged under CLI overrides) |
| `overrides` / `sweep.<flag>` | none | job grids — see below |

Use `dry=true` to inspect the generated script before committing GPUs.

## Job grids (sweeps)

Two ways to fan a single `submit` call into many jobs (combined as a **Cartesian
product** via `itertools.product`):

- **`sweep.<flag>=[v1,v2,…]`** (inline) or **`sweep.<flag>=values.yaml`** (a YAML
  list). One job per value; each appends **`<flag>=<value>`** to the command.
- **`overrides=grid.yaml`** — a YAML list (one axis of literal override strings) or
  a dict `{flag: [v1,v2]}` (expanded to `flag=v` strings, one axis per key).

> **Critical — appended tokens are OmegaConf `key=value` (no `--`).** This matches
> this repo's `apps/*.py`, which use `CliConfig`/`TaskConfig.from_cli()` and take
> `key=value` args. It does **not** work for `argparse` scripts (which need
> `--flag value`); for those, loop `submit` per value instead (see legacy example).

## Examples

Train (the launcher's default `script_with_args` is `apps/train.py`):
```bash
mkdir -p slurm   # once
submit apps/train.py config/my_train.yaml -- gpu_num=4 time=12 memory=128 job_name=delphi-train
```

Eval — a `CliConfig` app takes `key=value` args (no `--` for the script's own args):
```bash
submit apps/c-index-m4.py ckpt=delphi-m4/delphi-m4/ckpt.pt offset=10 -- time=2 job_name=cindex
```

Sweep a `CliConfig` app → one job per offset (5 jobs):
```bash
submit apps/c-index-m4.py ckpt=delphi-m4/delphi-m4/ckpt.pt -- "sweep.offset=[0,1,2,5,10]" time=1
```

Preview without submitting:
```bash
submit apps/c-index-m4.py ckpt=… -- "sweep.offset=[0,5,10]" dry=true
```

Legacy / `argparse` script (`--flag value`) — sweeps don't apply, loop instead:
```bash
for Y in 0 1 5 10; do
  submit some_argparse_script.py --offset_years $Y --out ./res/off$Y -- time=1 job_name=run_off$Y
done
```

## Gotchas (each cost a real debugging cycle)

1. **No leading `python`** — the template prepends it.
2. **`.env` + `slurm/` must exist in the submission CWD** (the repo root has the
   `.env`; `mkdir -p slurm`).
3. **Sweeps emit `flag=value`** (OmegaConf) → great for `apps/*.py`, wrong for
   `argparse` scripts → loop those.
4. **`dry=true`** writes/keeps the script in `/dev/shm` and prints it — use it to
   sanity-check the `python …` line and `#SBATCH` headers first.
5. `block_size`-style numeric flags: the script must actually expose them; the
   launcher only forwards args, it doesn't know your script's interface.

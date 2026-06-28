import os
from pathlib import Path

# assumes this file lives at <repo_root>/delphi/env.py
_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env_file() -> None:
    """Load KEY=VALUE pairs from <repo_root>/.env into os.environ.

    os.environ wins on conflict (explicit env vars override the file).
    Tolerates `export KEY=VALUE` and `KEY=VALUE`, comments, and quoted values.
    """
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file()
DELPHI_DATASET = os.environ.get("DELPHI_DATASET", "")
# default logging backend (Logger / TrainBaseConfig.log_backend); validated there.
# Set in the workbench .env so AoU runs default to trackio; unset -> wandb.
DELPHI_LOG_BACKEND = os.environ.get("DELPHI_LOG_BACKEND", "wandb")


dx_id = os.getenv("DX_PROJECT_CONTEXT_ID")
IN_RAP = dx_id != None

DELPHI_DATA_READ = os.environ.get("DELPHI_DATA_DIR", "/mnt/project/data")
DELPHI_DATA_WRITE = os.environ.get("DELPHI_DATA_DIR", "/opt/data")
DELPHI_DATA_DIR = DELPHI_DATA_READ

_ckpt_dir = os.environ.get("DELPHI_CKPT_DIR")
DELPHI_CKPT_READ = os.environ.get("DELPHI_CKPT_READ", _ckpt_dir or "/mnt/project/ckpt")
DELPHI_CKPT_WRITE = os.environ.get("DELPHI_CKPT_WRITE", _ckpt_dir or "/tmp/ckpt")
DELPHI_CKPT_DIR = _ckpt_dir or DELPHI_CKPT_WRITE

DELPHI_RESULTS_DIR = os.environ.get("DELPHI_RESULTS_DIR", str(_REPO_ROOT / "results"))

import os

DELPHI_DATA_DIR = os.environ.get("DELPHI_DATA_DIR", "/opt/data")

DELPHI_CKPT_READ = os.environ.get("DELPHI_CKPT_READ", "/mnt/project/ckpt")
DELPHI_CKPT_WRITE = os.environ.get("DELPHI_CKPT_WRITE", "/tmp/ckpt")
DELPHI_CKPT_DIR = DELPHI_CKPT_WRITE

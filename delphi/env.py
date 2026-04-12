import os

dx_id = os.getenv("DX_PROJECT_CONTEXT_ID")
IN_RAP = dx_id != None

DELPHI_DATA_READ = os.environ.get("DELPHI_DATA_DIR", "/mnt/project/data")
DELPHI_DATA_WRITE = os.environ.get("DELPHI_DATA_DIR", "/opt/data")
DELPHI_DATA_DIR = DELPHI_DATA_READ

DELPHI_CKPT_READ = os.environ.get("DELPHI_CKPT_READ", "/mnt/project/ckpt")
DELPHI_CKPT_WRITE = os.environ.get("DELPHI_CKPT_WRITE", "/tmp/ckpt")
DELPHI_CKPT_DIR = DELPHI_CKPT_WRITE

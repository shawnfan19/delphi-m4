import os

dx_id = os.getenv("DX_PROJECT_CONTEXT_ID")
IN_RAP = dx_id != None

if IN_RAP:
    DELPHI_DATA_DIR = "opt/notebooks/data"
    DELPHI_CKPT_DIR = "opt/notebooks/ckpt"
else:
    try:
        DELPHI_DATA_DIR = os.environ["DELPHI_DATA_DIR"]
        DELPHI_CKPT_DIR = os.environ["DELPHI_CKPT_DIR"]
    except KeyError as e:
        raise EnvironmentError(f"required environment variable(s) not set: {e}")

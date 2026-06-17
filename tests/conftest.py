import os
from pathlib import Path


os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("ENV_FOLDER", str(Path(__file__).parent))

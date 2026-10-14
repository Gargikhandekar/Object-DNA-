"""Register the synthetic demo dataset's reference images in the app (for a quick UI tour).

    python scripts/seed_demo.py
"""
import _bootstrap  # noqa: F401

import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings

if __name__ == "__main__":
    ds = get_settings().data_dir / "datasets" / "synthetic_demo"
    here = Path(__file__).parent
    if not (ds / "manifest.csv").exists():
        subprocess.check_call([sys.executable, str(here / "generate_synthetic_dataset.py")])
    subprocess.check_call([sys.executable, str(here / "import_dataset.py"), str(ds), "--register"])

"""Delete ALL registered objects, images, fingerprints and history (keeps datasets, weights, experiments).

    python scripts/reset_app_data.py          # asks for confirmation
    python scripts/reset_app_data.py --yes

Stop the backend first. Afterwards re-register what you want, e.g.
    python scripts/download_coil100.py --register
"""
import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path

from app.core.config import get_settings
from app.database import init_db

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    db = s.data_dir / "object_dna.db"
    storage = s.resolved_storage_dir
    print(f"This deletes {db} and {storage} (all registered objects and identification history).")
    if not a.yes and input("Type 'delete' to continue: ").strip() != "delete":
        raise SystemExit("Cancelled.")
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    shutil.rmtree(storage, ignore_errors=True)
    s.ensure_dirs()
    init_db()
    print("Done. The app now has no registered objects.")

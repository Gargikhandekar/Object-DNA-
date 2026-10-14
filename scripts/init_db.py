"""Create the SQLite database and storage folders.

    python scripts/init_db.py
"""
import _bootstrap  # noqa: F401

from app.core.config import get_settings
from app.database import init_db

if __name__ == "__main__":
    s = get_settings()
    s.ensure_dirs()
    init_db()
    print(f"Database ready: {s.resolved_database_url}")
    print(f"Storage dir:    {s.resolved_storage_dir}")

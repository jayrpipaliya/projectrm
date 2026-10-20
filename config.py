import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("MASSIVE_API_KEY", "").strip() or os.getenv("POLYGON_API_KEY", "").strip()
BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.polygon.io").strip().rstrip("/")
if os.getenv("VERCEL"):
    import shutil
    tmp_db = Path("/tmp") / "warehouse.db"
    orig_db = BASE_DIR / "data" / "warehouse.db"
    if not tmp_db.exists() and orig_db.exists():
        shutil.copy2(orig_db, tmp_db)
    DATABASE_PATH = tmp_db
else:
    DATABASE_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "data/warehouse.db")
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

PORT = int(os.getenv("PORT", "8050"))


def update_api_key(new_key: str):
    """Update API key in memory and in the .env file."""
    global API_KEY
    API_KEY = new_key.strip()

    env_path = BASE_DIR / ".env"
    lines = []
    found = False

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MASSIVE_API_KEY="):
                    lines.append(f"MASSIVE_API_KEY={API_KEY}\n")
                    found = True
                else:
                    lines.append(line)

    if not found:
        lines.append(f"MASSIVE_API_KEY={API_KEY}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

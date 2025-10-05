# python -m pip install pathlib2
# run: python scripts/reset_onboarding.py
import json
import os
from pathlib import Path
import sqlite3

# locations to try (dev project and per-user appdata)
candidates = [
    Path("src/config/settings.json"),
    Path(os.getenv("APPDATA") or "") / "TableTennisRideShare" / "settings.json",
]
db_candidates = [
    Path("src/rideshare.db"),
    Path(os.getenv("APPDATA") or "") / "TableTennisRideShare" / "rideshare.db",
]


def reset_settings(path: Path):
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.setdefault("onboarding", {})["completed"] = False
    # optionally clear stored API key for testing:
    # data["google_maps_api_key"] = ""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {path}")


def wipe_db(path: Path):
    if not path.exists():
        return
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    # WARNING: these delete all rides/team/ledger — keep a backup if needed
    for tbl in ("rides", "ledger", "team_members", "ride_drivers"):
        try:
            cur.execute(f"DELETE FROM {tbl};")
            print(f"Cleared {tbl} in {path}")
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()


for p in candidates:
    reset_settings(p)

for db in db_candidates:
    wipe_db(db)

print("Reset complete. Start the app; onboarding should run again.")

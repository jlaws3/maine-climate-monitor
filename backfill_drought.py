
from datetime import date, timedelta
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

START = date(2020, 1, 7)
END = date(2026, 10, 20)

dates = []
current = START

while current <= END:
    dates.append(current.isoformat())
    current += timedelta(days=7)

print(f"Historical drought weeks: {len(dates)}", flush=True)

for number, observation in enumerate(dates, 1):
    folder = ROOT / "data" / "drought-history" / observation

    geojson = folder / "drought.geojson"
    summary = folder / "drought-summary.json"

    if geojson.exists() and summary.exists():
        print(
            f"[{number}/{len(dates)}] SKIP {observation}",
            flush=True
        )
        continue

    print(
        f"[{number}/{len(dates)}] PROCESS {observation}",
        flush=True
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "process_drought_history.py"),
            observation
        ],
        cwd=ROOT
    )

    if result.returncode != 0:
        print(f"FAILED: {observation}", flush=True)
        sys.exit(result.returncode)

    time.sleep(1)

print("Historical drought backfill complete.", flush=True)

"""Run every Genesis/AMD mission gate and preserve per-test logs."""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MISSIONS = (
    "echos_avoid.py",
    "echos_frontier.py",
    "echos_flight.py",
    "echos_mapper.py",
)


def run_mission(script: str, log_dir: Path) -> int:
    log_path = log_dir / f"{Path(script).stem}.log"
    print(f"\n{'=' * 72}")
    print(f"RUN {script}")
    print(f"LOG {log_path}")
    print(f"{'=' * 72}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def main() -> int:
    missing = [script for script in MISSIONS if not Path(script).is_file()]
    if missing:
        print(f"Missing mission scripts: {', '.join(missing)}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = Path("gpu-validation-logs") / stamp
    log_dir.mkdir(parents=True, exist_ok=False)

    results = {}
    for script in MISSIONS:
        results[script] = run_mission(script, log_dir)

    print(f"\n{'=' * 72}")
    print("GPU VALIDATION SUMMARY")
    print(f"{'=' * 72}")
    for script, return_code in results.items():
        status = "PASS" if return_code == 0 else f"FAIL ({return_code})"
        print(f"{script:24s} {status}")
    print(f"Logs: {log_dir}")

    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

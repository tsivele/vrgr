#!/usr/bin/env python3
"""
VRGR — εκκίνηση εφαρμογής.

    python3 app.py            ανοίγει στο http://127.0.0.1:8778
    python3 app.py --port 9000
    python3 app.py --no-browser
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vrgr.config import load_settings                    # noqa: E402
from vrgr.logging_setup import setup_logging             # noqa: E402
from vrgr.web.server import serve                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="VRGR — Viral Intelligence (GR)")
    ap.add_argument("--port", type=int, default=8778)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--log-level", default="")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(args.log_level or settings.log_level, settings.log_format,
                  settings.data_dir / "vrgr.log")

    if not settings.models.enabled:
        print("\n  ⚠ Λείπει το ANTHROPIC_API_KEY από το .env — η ανάλυση δεν θα δουλέψει.")
    if not settings.hiker.enabled:
        print("  ⚠ Λείπει το HIKER_API_KEY από το .env — δεν θα γίνεται έρευνα δεδομένων.")

    serve(settings, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())

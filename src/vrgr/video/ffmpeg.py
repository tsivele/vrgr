"""
Εντοπισμός και εκτέλεση ffmpeg.

Σχεδιαστική απόφαση: ΔΕΝ απαιτούμε system ffmpeg. Το πακέτο pip
`imageio-ffmpeg` κουβαλά στατικό binary, οπότε η εγκατάσταση δουλεύει
σε μηχάνημα χωρίς Homebrew/Docker — ακριβώς η περίπτωση αυτού του συστήματος.

Το `imageio-ffmpeg` ΔΕΝ φέρνει ffprobe, γι' αυτό το probe.py ξέρει να
διαβάζει μεταδεδομένα και από το stderr του ίδιου του ffmpeg.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache
from typing import Optional

from ..errors import FFmpegMissing
from ..logging_setup import get_logger

log = get_logger("ffmpeg")


@lru_cache(maxsize=1)
def ffmpeg_path() -> str:
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg                    # noqa: PLC0415
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                            # noqa: BLE001
        raise FFmpegMissing(
            "Δεν βρέθηκε ffmpeg. Εγκατάστησέ το με:\n"
            "    python3 -m pip install --user imageio-ffmpeg\n"
            "(δεν χρειάζεται Homebrew ή Docker)")


@lru_cache(maxsize=1)
def ffprobe_path() -> Optional[str]:
    return shutil.which("ffprobe")


def run(args: list, timeout: int = 180) -> subprocess.CompletedProcess:
    """Εκτελεί ffmpeg. Το ffmpeg γράφει τα πάντα στο stderr — φυσιολογικό."""
    cmd = [ffmpeg_path(), "-hide_banner", "-nostdin", "-loglevel", "info"] + args
    log.debug("ffmpeg %s", " ".join(args[:12]))
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout, errors="replace")


def run_probe(args: list, timeout: int = 60) -> Optional[str]:
    path = ffprobe_path()
    if not path:
        return None
    proc = subprocess.run([path, "-hide_banner", "-loglevel", "error"] + args,
                          capture_output=True, text=True, timeout=timeout,
                          errors="replace")
    return proc.stdout if proc.returncode == 0 else None


def available() -> dict:
    try:
        fm = ffmpeg_path()
    except FFmpegMissing:
        fm = ""
    return {"ffmpeg": fm, "ffprobe": ffprobe_path() or "",
            "ok": bool(fm)}

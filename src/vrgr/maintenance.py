"""
Συντήρηση δίσκου.

Χωρίς αυτό το module το σύστημα μεγαλώνει για πάντα:

  • `hiker_cache.db` έφτασε 80 MB μετά από λίγες εκτελέσεις. Η `purge_expired`
    υπήρχε αλλά δεν την καλούσε κανείς.
  • Κάθε ανάλυση αφήνει 10-14 JPEG καρέ σε δικό της φάκελο. Δεν χρειάζονται
    μετά την ολοκλήρωση — η ανάλυση είναι ήδη αποθηκευμένη ως JSON.
  • Ένα αποτυχημένο ανέβασμα αφήνει το βίντεο στον δίσκο.

Ό,τι ΔΕΝ σβήνεται ποτέ: η βάση `vrgr.db` (posts, στιγμιότυπα, μοτίβα,
εκτελέσεις, αποτελέσματα). Αυτή είναι η γνώση του συστήματος και μεγαλώνει
σκόπιμα — 15 MB για 1.150 posts, δηλαδή ~1 GB για 75.000 posts.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

from .config import Settings
from .logging_setup import get_logger

log = get_logger("maintenance")

FRAMES_KEEP_DAYS = 3.0
UPLOADS_KEEP_HOURS = 6.0
CACHE_KEEP_DAYS = 21.0
RUNS_JSON_KEEP_DAYS = 90.0


def _age_days(path: Path) -> float:
    try:
        return (time.time() - path.stat().st_mtime) / 86400.0
    except OSError:
        return 0.0


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def cleanup(settings: Settings, cache=None, aggressive: bool = False) -> dict:
    """Επιστρέφει τι ελευθερώθηκε. Ασφαλές να τρέχει οποτεδήποτε."""
    freed = {"frames_dirs": 0, "uploads": 0, "cache_rows": 0,
             "runs_json": 0, "bytes": 0}
    factor = 0.25 if aggressive else 1.0

    # Καρέ παλιών εκτελέσεων
    media = settings.media_dir
    if media.is_dir():
        for d in media.iterdir():
            if not d.is_dir():
                continue
            if _age_days(d) > FRAMES_KEEP_DAYS * factor:
                size = _dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
                freed["frames_dirs"] += 1
                freed["bytes"] += size

    # Ανεβασμένα βίντεο — διαγράφονται από την εργασία, αλλά μια αποτυχία
    # πριν την έναρξη αφήνει ορφανά.
    uploads = settings.data_dir / "uploads"
    if uploads.is_dir():
        for f in uploads.iterdir():
            if not f.is_file():
                continue
            if _age_days(f) * 24 > UPLOADS_KEEP_HOURS * factor:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    freed["uploads"] += 1
                    freed["bytes"] += size
                except OSError:
                    continue

    # Ληγμένες εγγραφές cache HikerAPI
    if cache is not None:
        try:
            before = settings.cache_path.stat().st_size
            freed["cache_rows"] = cache.purge_expired(CACHE_KEEP_DAYS * factor)
            if freed["cache_rows"]:
                cache.vacuum()
                freed["bytes"] += max(0, before - settings.cache_path.stat().st_size)
        except Exception as exc:                     # noqa: BLE001
            log.warning("Καθαρισμός cache απέτυχε: %s", type(exc).__name__)

    # Εξαγωγές JSON — τα αποτελέσματα ζουν ήδη στη βάση
    runs = settings.runs_dir
    if runs.is_dir():
        for f in runs.glob("*.json"):
            if _age_days(f) > RUNS_JSON_KEEP_DAYS * factor:
                try:
                    size = f.stat().st_size
                    f.unlink()
                    freed["runs_json"] += 1
                    freed["bytes"] += size
                except OSError:
                    continue

    if freed["bytes"]:
        log.info("Συντήρηση: ελευθερώθηκαν %.1f MB (%d φάκελοι καρέ, %d uploads, "
                 "%d εγγραφές cache)", freed["bytes"] / 1e6, freed["frames_dirs"],
                 freed["uploads"], freed["cache_rows"])
    return freed


def usage(settings: Settings) -> dict:
    """Τι καταλαμβάνει χώρο — για την εντολή doctor και τη διεπαφή."""
    def size(p: Path) -> int:
        if not p.exists():
            return 0
        return p.stat().st_size if p.is_file() else _dir_size(p)

    return {
        "βάση_γνώσης": size(settings.db_path),
        "cache_hikerapi": size(settings.cache_path),
        "καρέ_βίντεο": size(settings.media_dir),
        "ανεβασμένα": size(settings.data_dir / "uploads"),
        "εξαγωγές_json": size(settings.runs_dir),
        "σύνολο": size(settings.data_dir),
    }

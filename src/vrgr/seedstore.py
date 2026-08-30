"""
Επιβίωση μνήμης σε εφήμερο filesystem (Streamlit Community Cloud).

ΤΟ ΠΡΟΒΛΗΜΑ: το Streamlit Cloud ξαναχτίζει το container από το GitHub σε κάθε
reboot, και οι εφαρμογές κοιμούνται μετά από 12 ώρες χωρίς επισκέψεις. Ό,τι
γράφεται στον δίσκο στην εκτέλεση χάνεται.

Η ΜΕΡΙΚΗ ΘΕΡΑΠΕΙΑ: μια βάση-αφετηρία (`seed/vrgr_seed.db`) ζει ΜΕΣΑ στο repo.
Σε κάθε ξεκίνημα, αν δεν υπάρχει βάση εργασίας, αντιγράφεται από εκεί. Έτσι
το σύστημα δεν ξεκινά ποτέ από το μηδέν — ξεκινά από όση γνώση είχε όταν
έγινε το τελευταίο commit.

ΤΙ ΔΕΝ ΛΥΝΕΙ: ό,τι μαθαίνει ΜΕΤΑ το τελευταίο commit χάνεται στο επόμενο
reboot. Γι' αυτό υπάρχει λήψη/επαναφορά αντιγράφου: κατεβάζεις τη βάση,
την κάνεις commit, και η γνώση γίνεται μόνιμη. Χειροκίνητο, αλλά λειτουργεί.

Η οριστική λύση είναι εξωτερική βάση (Postgres) — δεν είναι ενεργοποιημένη.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from .logging_setup import get_logger

log = get_logger("seedstore")

SEED_DIR = Path(__file__).resolve().parents[2] / "seed"
SEED_DB = SEED_DIR / "vrgr_seed.db"


def is_ephemeral() -> bool:
    """
    Τρέχουμε σε περιβάλλον που χάνει τον δίσκο;

    Το Streamlit Cloud δεν έχει επίσημη μεταβλητή· ελέγχουμε τα ίχνη που
    αφήνει το περιβάλλον του. Σε αμφιβολία επιστρέφουμε False, ώστε ένα
    τοπικό μηχάνημα να μη δει ποτέ λάθος προειδοποίηση.
    """
    markers = ("STREAMLIT_SHARING_MODE", "STREAMLIT_SERVER_HEADLESS_CLOUD",
               "STREAMLIT_RUNTIME_ENV")
    if any(os.environ.get(m) for m in markers):
        return True
    # Το Community Cloud τρέχει τα apps κάτω από /mount/src ή /app.
    here = str(Path(__file__).resolve())
    return here.startswith("/mount/src") or here.startswith("/app/")


def restore_if_missing(db_path: Path) -> Optional[str]:
    """
    Αν λείπει η βάση εργασίας, την ξεκινά από το seed του repo.

    Επιστρέφει μήνυμα για τη διεπαφή, ή None αν δεν έγινε τίποτα.
    """
    if db_path.exists() and db_path.stat().st_size > 0:
        return None
    if not SEED_DB.is_file():
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SEED_DB, db_path)
    size = db_path.stat().st_size / 1e6
    log.info("Η μνήμη αποκαταστάθηκε από το seed του repo (%.1f MB)", size)
    return (f"Η μνήμη ξεκίνησε από το αντίγραφο του repo ({size:.0f} MB). "
            f"Ό,τι μαθευτεί τώρα χάνεται στο επόμενο restart — "
            f"κατέβασε αντίγραφο και κάν' το commit για να μείνει.")


def export_db(db_path: Path, dest: Path) -> Path:
    """
    Συνεπές αντίγραφο της βάσης, ακόμη και ενώ γράφεται.

    Το `sqlite3.backup` παίρνει το κλείδωμα σωστά· απλή αντιγραφή αρχείου με
    ενεργό WAL μπορεί να δώσει κατεστραμμένο αντίγραφο.
    """
    import sqlite3
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def import_db(uploaded_bytes: bytes, db_path: Path) -> str:
    """Επαναφορά από ανεβασμένο αντίγραφο, με έλεγχο εγκυρότητας."""
    import sqlite3
    import tempfile
    tmp = Path(tempfile.mkstemp(suffix=".db")[1])
    tmp.write_bytes(uploaded_bytes)
    try:
        conn = sqlite3.connect(str(tmp))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        counts = {}
        for t in ("posts", "patterns", "runs"):
            if t in tables:
                counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        conn.close()
        required = {"posts", "post_snapshots", "patterns", "runs"}
        if not required.issubset(tables):
            raise ValueError("Το αρχείο δεν είναι βάση VRGR "
                             f"(λείπουν: {sorted(required - tables)})")
        # Καθαρίζουμε τα WAL/SHM του παλιού, αλλιώς μένουν ασύμβατα
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_path) + suffix)
            if side.exists():
                side.unlink()
        shutil.move(str(tmp), str(db_path))
        return ("Η μνήμη αποκαταστάθηκε: "
                + " · ".join(f"{v} {k}" for k, v in counts.items()))
    finally:
        if tmp.exists():
            tmp.unlink()

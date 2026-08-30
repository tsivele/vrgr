"""
Ήχος: εξαγωγή + προαιρετική μεταγραφή.

ΕΙΛΙΚΡΙΝΗΣ ΠΕΡΙΟΡΙΣΜΟΣ: η Anthropic δεν δέχεται ήχο. Χωρίς ρυθμισμένο
πάροχο ASR το σύστημα ΔΕΝ μαντεύει τι λέγεται — επιστρέφει κενή μεταγραφή
και το καταγράφει ως κενό δεδομένων στο report. Το να «φανταστεί» διάλογο
θα ήταν ακριβώς η κατασκευή δεδομένων που απαγορεύεται.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import VideoConfig
from ..logging_setup import get_logger
from . import ffmpeg as FF

log = get_logger("video.audio")


def extract_audio(path: Path, out_dir: Path) -> Optional[Path]:
    """Mono 16 kHz — το φορμά που θέλουν όλοι οι ASR πάροχοι."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "audio.m4a"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    proc = FF.run(["-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                   "-c:a", "aac", "-b:a", "64k", "-y", str(dest)], timeout=180)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    log.info("Χωρίς κομμάτι ήχου: %s", (proc.stderr or "")[-140:])
    return None


def transcribe(audio_path: Path, cfg: VideoConfig) -> dict:
    """
    Μεταγραφή. Επιστρέφει `{text, language, provider, available}`.

    `available=False` σημαίνει «δεν ξέρουμε τι λέγεται» — ΟΧΙ «δεν λέγεται τίποτα».
    """
    provider = (cfg.asr_provider or "none").lower()
    if provider == "none":
        return {"text": "", "language": "", "provider": "none", "available": False,
                "note": "Δεν έχει ρυθμιστεί πάροχος ASR (VRGR_ASR_PROVIDER). "
                        "Η ανάλυση βασίζεται μόνο σε καρέ."}
    try:
        if provider == "groq" and cfg.groq_key:
            return _groq(audio_path, cfg.groq_key)
        if provider == "openai" and cfg.openai_key:
            return _openai(audio_path, cfg.openai_key)
    except Exception as exc:                      # noqa: BLE001
        log.warning("Η μεταγραφή απέτυχε (%s)", type(exc).__name__)
        return {"text": "", "language": "", "provider": provider,
                "available": False, "note": f"Αποτυχία ASR: {type(exc).__name__}"}
    return {"text": "", "language": "", "provider": provider, "available": False,
            "note": f"Λείπει κλειδί για τον πάροχο ASR «{provider}»."}


def _groq(audio_path: Path, api_key: str) -> dict:
    import httpx                                   # noqa: PLC0415
    with open(audio_path, "rb") as fh:
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, fh, "audio/m4a")},
            data={"model": "whisper-large-v3", "language": "el",
                  "response_format": "json"},
            timeout=180)
    r.raise_for_status()
    return {"text": (r.json().get("text") or "").strip(), "language": "el",
            "provider": "groq", "available": True}


def _openai(audio_path: Path, api_key: str) -> dict:
    import httpx                                   # noqa: PLC0415
    with open(audio_path, "rb") as fh:
        r = httpx.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (audio_path.name, fh, "audio/m4a")},
            data={"model": "whisper-1", "language": "el"},
            timeout=180)
    r.raise_for_status()
    return {"text": (r.json().get("text") or "").strip(), "language": "el",
            "provider": "openai", "available": True}

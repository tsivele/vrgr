"""
Δειγματοληψία καρέ.

Κρίσιμη επιλογή: ΜΗ ομοιόμορφη δειγματοληψία.

Στα Reels η μάχη κρίνεται στα πρώτα ~3 δευτερόλεπτα — εκεί ο χρήστης
αποφασίζει αν θα μείνει. Ομοιόμορφη δειγματοληψία 10 καρέ σε βίντεο 30
δευτερολέπτων δίνει ΕΝΑ καρέ στο hook. Εμείς δίνουμε ~40% του προϋπολογισμού
καρέ στο παράθυρο hook και τα υπόλοιπα στην αφήγηση.

Επιπλέον προτιμώνται τα σημεία ΜΕΤΑ από αλλαγή σκηνής: ένα καρέ στη μέση
ενός πλάνου δείχνει το ίδιο με το προηγούμενο· ένα καρέ αμέσως μετά από
κόψιμο δείχνει κάτι νέο.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from ..logging_setup import get_logger
from ..schemas import VideoTechnical
from . import ffmpeg as FF

log = get_logger("video.frames")

HOOK_BUDGET_SHARE = 0.40


def plan_timestamps(tech: VideoTechnical, max_frames: int = 14,
                    hook_window_s: float = 3.0) -> list:
    """Ποιες χρονικές στιγμές αξίζει να δούμε."""
    dur = tech.duration_s
    if dur <= 0:
        return [0.0]
    if dur <= hook_window_s * 1.5:
        step = dur / (max_frames + 1)
        return [round(step * (i + 1), 2) for i in range(max_frames)]

    hook_n = max(2, int(round(max_frames * HOOK_BUDGET_SHARE)))
    rest_n = max(1, max_frames - hook_n)

    hook_end = min(hook_window_s, dur * 0.35)
    hook_ts = [round(hook_end * (i + 0.5) / hook_n, 2) for i in range(hook_n)]

    body_start, body_end = hook_end, max(hook_end + 0.5, dur - 0.35)
    span = body_end - body_start
    body_ts = [round(body_start + span * (i + 0.5) / rest_n, 2) for i in range(rest_n)]

    # Ευθυγράμμιση με κοψίματα: μετακίνηση σε 0.25s μετά το πλησιέστερο κόψιμο.
    cuts = list(tech.scene_cuts or [])
    if cuts:
        aligned = []
        for t in body_ts:
            near = min(cuts, key=lambda c: abs(c - t))
            aligned.append(round(near + 0.25, 2) if abs(near - t) < span / rest_n
                           else t)
        body_ts = aligned

    ts = sorted({t for t in hook_ts + body_ts if 0 <= t < dur})
    return ts[:max_frames]


def extract(path: Path, timestamps: list, out_dir: Path,
            width: int = 768) -> list:
    """
    Εξαγωγή JPEG ανά χρονική στιγμή.

    Ξεχωριστή κλήση ανά καρέ με `-ss` ΠΡΙΝ το `-i` (fast seek): πολύ
    ταχύτερο από σάρωση όλου του βίντεο και ακριβές αρκετά για ανάλυση.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(timestamps):
        dest = out_dir / f"frame_{i:02d}_{t:07.2f}.jpg"
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(str(dest))
            continue
        proc = FF.run(["-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                       "-vf", f"scale={width}:-2:flags=bicubic",
                       "-q:v", "3", "-y", str(dest)], timeout=90)
        if dest.exists() and dest.stat().st_size > 0:
            paths.append(str(dest))
        else:
            log.warning("Απέτυχε η εξαγωγή καρέ στο %.2fs: %s",
                        t, (proc.stderr or "")[-160:])
    return paths


def to_base64(frame_path: str, max_bytes: int = 4_500_000) -> Optional[dict]:
    """Καρέ → block εικόνας για το Anthropic API."""
    p = Path(frame_path)
    if not p.is_file():
        return None
    data = p.read_bytes()
    if len(data) > max_bytes:
        try:
            from PIL import Image                 # noqa: PLC0415
            import io
            img = Image.open(io.BytesIO(data))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=72, optimize=True)
            data = buf.getvalue()
        except Exception:                         # noqa: BLE001
            return None
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": base64.standard_b64encode(data).decode("ascii")},
    }


def sample(path: Path, tech: VideoTechnical, out_dir: Path,
           max_frames: int = 14, hook_window_s: float = 3.0,
           width: int = 768) -> VideoTechnical:
    ts = plan_timestamps(tech, max_frames, hook_window_s)
    tech.frame_paths = extract(path, ts, out_dir, width)
    tech.frame_times = ts[:len(tech.frame_paths)]
    log.info("Εξήχθησαν %d καρέ (%d στο hook ≤%.1fs)",
             len(tech.frame_paths),
             sum(1 for t in tech.frame_times if t <= hook_window_s), hook_window_s)
    return tech

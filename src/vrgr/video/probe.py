"""Τεχνικά χαρακτηριστικά βίντεο — ντετερμινιστικά, χωρίς LLM."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Optional

from ..errors import VideoError
from ..logging_setup import get_logger
from ..schemas import VideoTechnical
from . import ffmpeg as FF

log = get_logger("video.probe")

_DURATION = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d\.\d+)")
_VIDEO_STREAM = re.compile(
    r"Stream #\d+:\d+.*?:\s*Video:.*?,\s*(\d+)x(\d+).*?,\s*([\d.]+)\s*fps", re.S)
_AUDIO_STREAM = re.compile(r"Stream #\d+:\d+.*?:\s*Audio:")
_SHOWINFO_PTS = re.compile(r"pts_time:([\d.]+)")


def _aspect(w: int, h: int) -> str:
    if not w or not h:
        return ""
    g = math.gcd(w, h)
    return f"{w // g}:{h // g}"


def probe(path: Path) -> VideoTechnical:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"Το αρχείο δεν βρέθηκε: {path}")
    size = path.stat().st_size
    if size == 0:
        raise VideoError(f"Το αρχείο είναι κενό: {path}")

    tech = _probe_ffprobe(path) or _probe_ffmpeg(path)
    tech.path = str(path)
    tech.size_bytes = size
    if tech.duration_s <= 0:
        raise VideoError(
            f"Αδύνατη ανάγνωση διάρκειας — πιθανώς κατεστραμμένο αρχείο: {path.name}")
    tech.aspect_ratio = _aspect(tech.width, tech.height)
    return tech


def _probe_ffprobe(path: Path) -> Optional[VideoTechnical]:
    out = FF.run_probe(["-print_format", "json", "-show_format",
                        "-show_streams", str(path)])
    if not out:
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        return None
    fps = 0.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0)
    return VideoTechnical(duration_s=duration, width=int(v.get("width") or 0),
                          height=int(v.get("height") or 0), fps=round(fps, 2),
                          has_audio=a is not None)


def _probe_ffmpeg(path: Path) -> VideoTechnical:
    """Fallback όταν λείπει ffprobe: διάβασμα του stderr του ffmpeg."""
    proc = FF.run(["-i", str(path), "-f", "null", "-"], timeout=120)
    err = proc.stderr or ""
    duration = 0.0
    m = _DURATION.search(err)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    w = h = 0
    fps = 0.0
    mv = _VIDEO_STREAM.search(err)
    if mv:
        w, h, fps = int(mv.group(1)), int(mv.group(2)), float(mv.group(3))
    return VideoTechnical(duration_s=duration, width=w, height=h,
                          fps=round(fps, 2),
                          has_audio=bool(_AUDIO_STREAM.search(err)))


def detect_scenes(path: Path, duration_s: float, threshold: float = 0.30,
                  hook_window_s: float = 3.0) -> dict:
    """
    Ανίχνευση αλλαγών σκηνής → ρυθμός μοντάζ.

    Ο ρυθμός κοψιμάτων ΣΤΑ ΠΡΩΤΑ ΔΕΥΤΕΡΟΛΕΠΤΑ είναι από τα ισχυρότερα
    τεχνικά σήματα hook: πυκνά κοψίματα στο άνοιγμα κρατούν προσοχή.
    Το μετράμε ντετερμινιστικά αντί να το ρωτήσουμε από μοντέλο.
    """
    proc = FF.run(["-i", str(path), "-filter:v",
                   f"select='gt(scene,{threshold})',showinfo",
                   "-f", "null", "-"], timeout=240)
    cuts = []
    for line in (proc.stderr or "").splitlines():
        if "showinfo" not in line:
            continue
        m = _SHOWINFO_PTS.search(line)
        if m:
            try:
                cuts.append(round(float(m.group(1)), 3))
            except ValueError:
                continue
    cuts = sorted(set(cuts))
    n = len(cuts)
    return {
        "scene_cuts": cuts,
        "cut_count": n,
        "cuts_per_second": round(n / duration_s, 3) if duration_s else 0.0,
        "avg_shot_len_s": round(duration_s / (n + 1), 2) if duration_s else None,
        "hook_cut_count": sum(1 for c in cuts if c <= hook_window_s),
    }


def full_probe(path: Path, threshold: float = 0.30,
               hook_window_s: float = 3.0) -> VideoTechnical:
    tech = probe(path)
    try:
        scenes = detect_scenes(path, tech.duration_s, threshold, hook_window_s)
        for key, value in scenes.items():
            setattr(tech, key, value)
    except Exception as exc:                      # noqa: BLE001
        log.warning("Ανίχνευση σκηνών απέτυχε (%s) — συνεχίζουμε χωρίς",
                    type(exc).__name__)
    return tech

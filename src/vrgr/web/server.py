"""
Τοπικός web server — μόνο βιβλιοθήκη Python, καμία εξάρτηση.

Γιατί stdlib αντί για FastAPI: το μηχάνημα δεν έχει brew/docker/node, και
κάθε επιπλέον πακέτο είναι ένα ακόμη σημείο αποτυχίας για μια εφαρμογή που
τρέχει τοπικά για έναν χρήστη. Το `ThreadingHTTPServer` καλύπτει άνετα την
περίπτωση, και η εγκατάσταση παραμένει «διπλό κλικ».

ΑΣΦΑΛΕΙΑ: δεσμεύεται ΜΟΝΟ στο 127.0.0.1. Ο server έχει πρόσβαση στα κλειδιά
σου και στη βάση· δεν πρέπει ποτέ να είναι προσβάσιμος από το δίκτυο.
"""
from __future__ import annotations

import json
import mimetypes
import re
import shutil
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from ..config import Settings
from ..logging_setup import get_logger
from .jobs import JobManager

log = get_logger("web")

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 400 * 1024 * 1024          # 400 MB — αρκετά για Reel 4K
ALLOWED_VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


# ── streaming parser multipart ────────────────────────────────────────
# Το `cgi.FieldStorage` είναι deprecated από την 3.11 και αφαιρέθηκε στην 3.13.
#
# ΚΑΙ ΤΟ ΚΥΡΙΟΤΕΡΟ: το προφανές `rfile.read(content_length)` φορτώνει ολόκληρο
# το βίντεο στη μνήμη. Για Reel 4K από κινητό (συχνά 150-300 MB) η κορύφωση
# έφτανε το ~1,2 GB μετά τα αντίγραφα του split — αρκετό για να σκοτώσει τη
# διεργασία σε μηχάνημα με 8 GB.
#
# Εδώ το σώμα διαβάζεται σε κομμάτια: τα μικρά πεδία μένουν στη μνήμη, το
# αρχείο γράφεται κατευθείαν στον δίσκο. Η κορύφωση είναι ~1 MB ανεξάρτητα
# από το μέγεθος του βίντεο.

CHUNK = 256 * 1024
MAX_FIELD = 1 * 1024 * 1024          # ένα πεδίο κειμένου δεν είναι ποτέ τόσο μεγάλο

_DISPOSITION = re.compile(
    r'name="(?P<name>[^"]*)"(?:;\s*filename="(?P<filename>[^"]*)")?')


class MultipartError(ValueError):
    pass


def _parse_headers(raw: bytes) -> tuple:
    """`(name, filename)` από τα headers ενός part."""
    m = _DISPOSITION.search(raw.decode("utf-8", "replace"))
    if not m:
        return None, None
    return m.group("name"), m.group("filename")


def stream_multipart(rfile, length: int, boundary: bytes, dest_dir: Path,
                     max_total: int) -> tuple:
    """
    Διαβάζει multipart σε ροή.

    Επιστρέφει `(fields, files)` όπου `files[name] = (filename, Path, bytes_written)`.
    Τα αρχεία γράφονται στον `dest_dir`· ο καλών είναι υπεύθυνος να τα σβήσει.
    """
    if length > max_total:
        raise MultipartError(
            f"Το αίτημα ξεπερνά τα {max_total // (1024*1024)} MB")

    delim = b"--" + boundary
    tail_keep = len(delim) + 8          # όσο χρειάζεται για boundary σε δύο κομμάτια
    fields: dict = {}
    files: dict = {}

    buf = b""
    remaining = length
    state = "seek"                      # seek → headers → body
    name = filename = None
    sink = None
    written = 0
    field_buf = b""

    def close_sink():
        nonlocal sink, written, name, filename
        if sink is not None:
            sink.close()
            files[name] = (filename, Path(sink.name), written)
            sink = None
            written = 0

    while True:
        if remaining > 0:
            chunk = rfile.read(min(CHUNK, remaining))
            if not chunk:
                remaining = 0
            else:
                remaining -= len(chunk)
                buf += chunk
        eof = remaining <= 0

        progressed = True
        while progressed:
            progressed = False

            if state == "seek":
                idx = buf.find(delim)
                if idx == -1:
                    buf = buf[-tail_keep:] if len(buf) > tail_keep else buf
                    break
                after = idx + len(delim)
                if buf[after:after + 2] == b"--":
                    return fields, files            # τερματικό boundary
                nl = buf.find(b"\r\n", after)
                if nl == -1:
                    if eof:
                        return fields, files
                    break
                buf = buf[nl + 2:]
                state = "headers"
                progressed = True

            elif state == "headers":
                end = buf.find(b"\r\n\r\n")
                if end == -1:
                    if eof:
                        raise MultipartError("Ημιτελή headers στο multipart")
                    break
                name, filename = _parse_headers(buf[:end])
                buf = buf[end + 4:]
                field_buf = b""
                if filename:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    sink = tempfile.NamedTemporaryFile(
                        dir=str(dest_dir), prefix="upload_", suffix=".part",
                        delete=False)
                    written = 0
                state = "body"
                progressed = True

            elif state == "body":
                idx = buf.find(b"\r\n" + delim)
                if idx == -1:
                    # Κρατάμε ουρά: το boundary μπορεί να κόβεται στα δύο.
                    if len(buf) > tail_keep:
                        emit, buf = buf[:-tail_keep], buf[-tail_keep:]
                        if sink is not None:
                            sink.write(emit)
                            written += len(emit)
                        else:
                            field_buf += emit
                            if len(field_buf) > MAX_FIELD:
                                raise MultipartError(
                                    f"Το πεδίο «{name}» είναι υπερβολικά μεγάλο")
                    if eof:
                        if sink is not None:
                            sink.write(buf)
                            written += len(buf)
                            close_sink()
                        elif name:
                            fields[name] = (field_buf + buf).decode("utf-8", "replace")
                        return fields, files
                    break
                emit, buf = buf[:idx], buf[idx + 2:]     # +2 = το \r\n πριν το boundary
                if sink is not None:
                    sink.write(emit)
                    written += len(emit)
                    close_sink()
                elif name:
                    fields[name] = (field_buf + emit).decode("utf-8", "replace")
                state = "seek"
                progressed = True

        if eof and not progressed:
            close_sink()
            return fields, files


class Handler(BaseHTTPRequestHandler):
    server_version = "VRGR"
    jobs: JobManager = None            # τίθεται από τον serve()
    settings: Settings = None

    # ── βοηθητικά ────────────────────────────────────────────────────
    def log_message(self, fmt, *args):
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              extra: Optional[dict] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Ο server δεν προορίζεται για ενσωμάτωση σε άλλη σελίδα.
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(code, json.dumps(data, ensure_ascii=False, default=str)
                   .encode("utf-8"))

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"error": message}, code)

    # ── δρομολόγηση ──────────────────────────────────────────────────
    def do_GET(self):                                   # noqa: N802
        url = urlparse(self.path)
        path, query = url.path, parse_qs(url.query)
        try:
            if path == "/" or path == "/index.html":
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/status":
                return self._json(self._status())
            if path == "/api/jobs":
                return self._json({"jobs": self.jobs.recent()})
            if path.startswith("/api/job/"):
                return self._job(path.rsplit("/", 1)[-1],
                                 want_result="result" in query)
            if path == "/api/runs":
                return self._json({"runs": self._runs(int(query.get("limit", [20])[0]))})
            if path.startswith("/api/run/"):
                return self._run_detail(path.rsplit("/", 1)[-1])
            if path == "/api/memory":
                return self._json(self._memory(query.get("q", [""])[0]))
            return self._error("Άγνωστη διαδρομή", 404)
        except Exception as exc:                        # noqa: BLE001
            log.error("GET %s: %s", path, exc)
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self):                                  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path == "/api/analyze":
                return self._analyze()
            if url.path == "/api/research":
                return self._research()
            if url.path == "/api/feedback":
                return self._feedback()
            return self._error("Άγνωστη διαδρομή", 404)
        except Exception as exc:                        # noqa: BLE001
            log.error("POST %s: %s", url.path, exc)
            return self._error(f"{type(exc).__name__}: {exc}", 500)

    # ── στατικά ──────────────────────────────────────────────────────
    def _static(self, rel: str) -> None:
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self._error("Δεν βρέθηκε", 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype == "application/javascript":
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    # ── endpoints ────────────────────────────────────────────────────
    def _status(self) -> dict:
        s = self.settings
        pipe = self.jobs.pipeline()
        stats = pipe.retriever.corpus_stats()
        patterns = pipe.patterns.stats()
        balance = None
        if pipe.hiker is not None:
            balance = pipe.hiker.balance()
        return {
            "keys": {"hiker": s.hiker.enabled, "anthropic": s.models.enabled},
            "models": {"vision": s.models.vision_model, "writer": s.models.writer_model},
            "embeddings": s.embeddings.provider,
            "asr": s.video.asr_provider,
            "memory": stats, "patterns": patterns,
            "balance": balance,
            "budget_per_run": s.hiker.budget_per_run,
            "disk": __import__("vrgr.maintenance", fromlist=["usage"]).usage(s),
        }

    def _job(self, job_id: str, want_result: bool) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return self._error("Άγνωστη εργασία", 404)
        payload = job.to_dict()
        if want_result and job.status == "done" and job.result is not None:
            if job.kind == "analyze":
                payload["result"] = _analysis_json(job.result)
            else:
                payload["result"] = job.result
        return self._json(payload)

    def _analyze(self) -> None:
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._error("Αναμένεται multipart/form-data")
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return self._error("Κενό αίτημα")
        if length > MAX_UPLOAD:
            return self._error(
                f"Το αρχείο ξεπερνά τα {MAX_UPLOAD // (1024*1024)} MB", 413)
        m = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", ctype)
        if not m:
            return self._error("Λείπει το boundary")
        boundary = (m.group(1) or m.group(2)).strip().encode()

        upload_dir = self.settings.data_dir / "uploads"
        try:
            fields, files = stream_multipart(self.rfile, length, boundary,
                                             upload_dir, MAX_UPLOAD)
        except MultipartError as exc:
            return self._error(str(exc), 413)

        def cleanup():
            for _, (_, path, _) in files.items():
                try:
                    path.unlink(missing_ok=True)
                except Exception:                    # noqa: BLE001
                    pass

        upload = files.get("video")
        if upload is None:
            cleanup()
            return self._error("Δεν στάλθηκε αρχείο βίντεο")
        filename, tmp_path, size = upload
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_VIDEO:
            cleanup()
            return self._error(
                f"Μη υποστηριζόμενος τύπος «{suffix}». Δεκτά: "
                + ", ".join(sorted(ALLOWED_VIDEO)))
        if size == 0:
            cleanup()
            return self._error("Το αρχείο είναι κενό")

        # Μετονομασία στη θέση του — καμία δεύτερη αντιγραφή στη μνήμη.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)[-80:]
        dest = upload_dir / f"{int(time.time())}_{safe}"
        tmp_path.replace(dest)

        creators = [c.strip().lstrip("@") for c in
                    (fields.get("creators") or "").split(",") if c.strip()]
        job = self.jobs.start_analysis(dest, {
            "context": fields.get("context", ""),
            "captions": fields.get("captions", "8"),
            "no_research": fields.get("no_research") in ("1", "true", "on"),
            "creators": creators or None,
        })
        log.info("Ανάλυση ξεκίνησε: %s (%.1f MB)", dest.name, size / 1e6)
        return self._json({"job_id": job.id}, 202)

    def _research(self) -> None:
        body = self._read_json()
        target = (body.get("target") or "").strip()
        if not target:
            return self._error("Δώσε @λογαριασμό ή #hashtag")
        job = self.jobs.start_research(target, int(body.get("budget") or 25))
        return self._json({"job_id": job.id}, 202)

    def _feedback(self) -> None:
        body = self._read_json()
        run_id = (body.get("run_id") or "").strip()
        if not run_id:
            return self._error("Λείπει το run_id")
        from ..learning.feedback import FeedbackLoop
        pipe = self.jobs.pipeline()
        loop = FeedbackLoop(pipe.repo, pipe.patterns, pipe.hiker)
        manual = {k: int(body[k]) for k in
                  ("views", "likes", "comments", "followers")
                  if body.get(k) not in (None, "", 0)}
        out = loop.record(run_id, body.get("url", ""), manual or None)
        out["summary"] = loop.summary()
        return self._json(out)

    def _runs(self, limit: int) -> list:
        return self.jobs.pipeline().repo.recent_runs(limit)

    def _run_detail(self, run_id: str) -> None:
        run = self.jobs.pipeline().repo.get_run(run_id)
        if not run:
            return self._error("Δεν βρέθηκε η εκτέλεση", 404)
        try:
            result = json.loads(run.pop("result_json", "{}") or "{}")
        except json.JSONDecodeError:
            result = {}
        # Το «έτοιμο για επικόλληση» παράγεται κατά την προβολή, ώστε παλιές
        # εκτελέσεις να ανοίγουν με την ίδια πληρότητα με τις καινούριες.
        w = result.get("winner") or {}
        if w:
            tags = (w.get("hashtag_set") or {}).get("tags") or []
            result["ready_to_paste"] = ((w.get("caption") or {}).get("text", "")
                                        + "\n\n" + " ".join("#" + t for t in tags))
        run["result"] = result
        return self._json(run)

    def _memory(self, query: str) -> dict:
        pipe = self.jobs.pipeline()
        out = {"stats": pipe.retriever.corpus_stats(),
               "patterns_stats": pipe.patterns.stats(), "results": [],
               "patterns": []}
        if query:
            out["results"] = pipe.retriever.search(query, limit=15)
        for kind in ("caption_structure", "hashtag"):
            for p in pipe.patterns.by_kind(kind, "", limit=10):
                out["patterns"].append({
                    "key": p.key, "kind": p.kind, "n": round(p.n, 1),
                    "mean": round(p.mean, 3), "lower": round(p.lower_bound(), 3),
                    "confidence": p.confidence, "description": p.description_el})
        return out

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}


def _analysis_json(result) -> dict:
    """Το `AnalysisResult` σε μορφή που θέλει η διεπαφή."""
    from ..report import render
    data = result.model_dump(mode="json")
    data["report_text"] = render(result, verbose=True)
    if result.winner:
        tags = result.winner.hashtag_set.tags
        data["ready_to_paste"] = (result.winner.caption.text + "\n\n"
                                  + " ".join(f"#{t}" for t in tags))
    return data


def _housekeeping(settings: Settings, jobs: JobManager) -> None:
    """Καθαρισμός στην εκκίνηση και μία φορά την ημέρα όσο τρέχει ο server."""
    from ..maintenance import cleanup
    try:
        pipe = jobs.pipeline()
        cleanup(settings, cache=getattr(pipe, "_cache", None))
    except Exception as exc:                          # noqa: BLE001
        log.warning("Η συντήρηση απέτυχε: %s", type(exc).__name__)
    t = threading.Timer(24 * 3600, _housekeeping, args=(settings, jobs))
    t.daemon = True
    t.start()


def serve(settings: Settings, port: int = 8778, open_browser: bool = True) -> None:
    Handler.jobs = JobManager(settings)
    Handler.settings = settings
    threading.Timer(3.0, _housekeeping, args=(settings, Handler.jobs)).start()
    # ΜΟΝΟ localhost: ο server βλέπει τα κλειδιά και τη βάση.
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"\n  VRGR — Instagram Reels Viral Intelligence")
    print(f"  ▸ {url}\n")
    print(f"  HikerAPI: {'✓' if settings.hiker.enabled else '✗ λείπει κλειδί'}"
          f"   Anthropic: {'✓' if settings.models.enabled else '✗ λείπει κλειδί'}")
    print(f"  Ctrl+C για τερματισμό\n")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Τερματισμός…")
    finally:
        httpd.server_close()
        Handler.jobs.close()

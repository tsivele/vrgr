"""
Logging με υποχρεωτικό redaction μυστικών.

Ο filter τρέχει πάνω σε ΚΑΘΕ record πριν τη μορφοποίηση, οπότε ακόμη κι αν
κάποιο module κατά λάθος βάλει ένα API key σε μήνυμα, δεν φτάνει ποτέ στο
αρχείο ή στο τερματικό.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Optional

from .config import secret_values

# Μοτίβα που μοιάζουν με κλειδί ακόμη κι αν δεν είναι στο environment.
_KEYLIKE = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(x-access-key\s*[:=]\s*)([A-Za-z0-9]{8,})"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)([A-Za-z0-9_\-]{8,})"),
]
_MASK = "***REDACTED***"


def redact(text: str) -> str:
    """Αφαιρεί μυστικά από οποιοδήποτε κείμενο."""
    if not text:
        return text
    for secret in secret_values():
        if secret in text:
            text = text.replace(secret, _MASK)
    for pat in _KEYLIKE:
        if pat.groups >= 2:
            text = pat.sub(lambda m: m.group(1) + _MASK, text)
        else:
            text = pat.sub(_MASK, text)
    return text


class _RedactFilter(logging.Filter):
    @staticmethod
    def _clean(value):
        """
        Καθαρίζει ΜΟΝΟ κείμενο.

        Κρίσιμο: τα μη-string args πρέπει να μείνουν στον τύπο τους. Αν
        μετατραπούν σε string, οι μορφοποιητές `%d` και `%.1f` του logging
        αποτυγχάνουν και το logging τυπώνει ολόκληρο traceback αντί για το
        μήνυμα — μετατρέποντας ένα μέτρο ασφαλείας σε πηγή θορύβου.
        """
        return redact(value) if isinstance(value, str) else value

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: self._clean(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(self._clean(a) for a in record.args)
        except Exception:            # ποτέ μη ρίξεις τη διεργασία για ένα log
            record.msg = "<σφάλμα redaction>"
            record.args = ()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("run_id", "endpoint", "step", "cost", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def setup_logging(level: str = "INFO", fmt: str = "text",
                  logfile: Optional[Any] = None) -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger("vrgr")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_RedactFilter())
    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-24s %(message)s", "%H:%M:%S"))
    root.addHandler(handler)

    if logfile:
        fh = logging.FileHandler(str(logfile), encoding="utf-8")
        fh.addFilter(_RedactFilter())
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"vrgr.{name}")

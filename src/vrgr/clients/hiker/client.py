"""
HikerAPI client.

Σχεδιαστικές αποφάσεις που δεν είναι προφανείς:

1. ΡΗΤΟ User-Agent. Το HikerAPI επιστρέφει 403 στο default UA της httpx.
   Επιβεβαιωμένο εμπειρικά (το ίδιο πρόβλημα λύθηκε με `curl` στο προηγούμενο
   project). Στέλνουμε `curl/8.7.1` και κρατάμε fallback σε πραγματικό curl.

2. ΜΗΤΡΩΟ endpoints. Ο client δέχεται μόνο `Endpoint` αντικείμενα από το
   επαληθευμένο registry — αδύνατο να κληθεί endpoint που δεν υπάρχει.

3. BUDGET ανά εκτέλεση. Κάθε κλήση κοστίζει credits· χωρίς σκληρό όριο μια
   λάθος παράμετρος μπορεί να αδειάσει τον λογαριασμό.

4. STALE-ON-ERROR. Σε 5xx/timeout προτιμάμε ληγμένα δεδομένα από cache
   (με ρητή σήμανση) παρά αποτυχία ολόκληρης της ανάλυσης.
"""
from __future__ import annotations

import json
import random
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ...cache.store import HttpCache
from ...config import HikerConfig
from ...errors import (BudgetExceeded, HikerAuthError, HikerError, HikerNotFound,
                       HikerRateLimited, HikerServerError)
from ...logging_setup import get_logger
from . import endpoints as E

log = get_logger("hiker")

try:
    import httpx
except ImportError:                      # pragma: no cover
    httpx = None


class _TokenBucket:
    """Ρυθμιστής ρυθμού. Μοιράζεται μεταξύ threads."""

    def __init__(self, rate_per_s: float, burst: Optional[float] = None):
        self.rate = max(0.1, rate_per_s)
        self.capacity = burst if burst is not None else max(1.0, rate_per_s)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity,
                               self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            wait = (1.0 - self._tokens) / self.rate
        time.sleep(wait)
        with self._lock:
            self._tokens = max(0.0, self._tokens - 1.0)
            self._last = time.monotonic()
        return wait


@dataclass
class ClientStats:
    calls: int = 0
    cache_hits: int = 0
    stale_hits: int = 0
    errors: int = 0
    retries: int = 0
    cost_units: int = 0
    total_latency_ms: float = 0.0
    per_endpoint: dict = field(default_factory=dict)

    def record(self, path: str, cached: bool, latency_ms: float,
               cost: int, error: bool = False) -> None:
        slot = self.per_endpoint.setdefault(
            path, {"calls": 0, "cache": 0, "errors": 0, "ms": 0.0})
        if cached:
            self.cache_hits += 1
            slot["cache"] += 1
            return
        self.calls += 1
        self.cost_units += cost
        self.total_latency_ms += latency_ms
        slot["calls"] += 1
        slot["ms"] += latency_ms
        if error:
            self.errors += 1
            slot["errors"] += 1

    def summary(self) -> dict:
        return {
            "api_calls": self.calls,
            "cache_hits": self.cache_hits,
            "stale_hits": self.stale_hits,
            "errors": self.errors,
            "retries": self.retries,
            "cost_units": self.cost_units,
            "avg_latency_ms": round(self.total_latency_ms / self.calls, 1)
            if self.calls else 0.0,
        }


class HikerClient:
    """Thread-safe client. Ένα instance ανά διεργασία αρκεί."""

    def __init__(self, cfg: HikerConfig, cache: Optional[HttpCache] = None,
                 budget: Optional[int] = None):
        self.cfg = cfg
        self.cache = cache
        self.stats = ClientStats()
        self._bucket = _TokenBucket(cfg.rate_limit_rps)
        self._budget = cfg.budget_per_run if budget is None else budget
        self._spent = 0
        self._lock = threading.Lock()
        self._base = cfg.base_url
        self._client = None
        if httpx is not None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(cfg.timeout_s),
                headers=self._headers(),
                follow_redirects=True,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )

    # ── εσωτερικά ────────────────────────────────────────────────────
    def _headers(self) -> dict:
        return {
            "accept": "application/json",
            "x-access-key": self.cfg.api_key,
            "User-Agent": self.cfg.user_agent,
        }

    def _ttl_seconds(self, family: str) -> float:
        return 60.0 * {
            "profile": self.cfg.ttl_profile_min,
            "media": self.cfg.ttl_media_min,
            "search": self.cfg.ttl_search_min,
            "hashtag": self.cfg.ttl_hashtag_min,
            "system": 5,
        }.get(family, self.cfg.ttl_search_min)

    def _check_budget(self, cost: int) -> None:
        with self._lock:
            if self._budget and self._spent + cost > self._budget:
                raise BudgetExceeded(
                    f"Εξαντλήθηκε το budget κλήσεων HikerAPI για αυτή την ανάλυση "
                    f"({self._spent}/{self._budget} μονάδες). Αύξησέ το με "
                    f"HIKER_BUDGET_PER_RUN αν χρειάζεται βαθύτερη έρευνα."
                )
            self._spent += cost

    @property
    def budget_left(self) -> int:
        with self._lock:
            return max(0, self._budget - self._spent) if self._budget else 10 ** 9

    def reset_budget(self, budget: Optional[int] = None) -> None:
        with self._lock:
            self._spent = 0
            if budget is not None:
                self._budget = budget

    # ── μεταφορά ─────────────────────────────────────────────────────
    def _raise_for_status(self, status: int, path: str, body: str) -> None:
        if status in (401, 403):
            raise HikerAuthError(
                "Το HikerAPI απέρριψε το κλειδί (401/403). Έλεγξε το HIKER_API_KEY "
                "ή αν έχει εξαντληθεί το πακέτο σου.", status, path, body)
        if status == 404:
            raise HikerNotFound("Δεν βρέθηκε (404) — ιδιωτικός/διαγραμμένος "
                                "λογαριασμός ή post.", status, path, body)
        if status == 429:
            raise HikerRateLimited("Rate limit (429).", status, path, body)
        if status >= 500:
            raise HikerServerError(f"Σφάλμα διακομιστή ({status}).",
                                   status, path, body)
        if status >= 400:
            raise HikerError(f"Σφάλμα HikerAPI ({status}).", status, path, body)

    def _request_httpx(self, path: str, params: dict) -> Any:
        assert self._client is not None
        r = self._client.get(self._base + path, params=params)
        self._raise_for_status(r.status_code, path, r.text)
        return r.json()

    def _request_curl(self, path: str, params: dict) -> Any:
        """Fallback όταν λείπει η httpx ή μπλοκάρεται το TLS fingerprint."""
        from urllib.parse import urlencode
        url = f"{self._base}{path}?{urlencode(params)}"
        proc = subprocess.run(
            ["curl", "-s", "-w", "\n%{http_code}", "--max-time",
             str(self.cfg.timeout_s), url,
             "-H", "accept: application/json",
             "-H", f"x-access-key: {self.cfg.api_key}"],
            capture_output=True, text=True,
        )
        out = proc.stdout or ""
        body, _, code = out.rpartition("\n")
        try:
            status = int(code.strip())
        except ValueError:
            raise HikerServerError("Άκυρη απόκριση από curl.", 0, path, out)
        self._raise_for_status(status, path, body)
        return json.loads(body) if body.strip() else {}

    # ── δημόσιο API ──────────────────────────────────────────────────
    def call(self, endpoint: E.Endpoint, use_cache: bool = True,
             **params: Any) -> Any:
        """
        Εκτελεί μία κλήση. Επιστρέφει το raw JSON του HikerAPI.

        Σφάλματα 404 δεν είναι αποτυχίες συστήματος — τα προωθούμε ώστε ο
        καλών να αποφασίσει (π.χ. ιδιωτικός λογαριασμός = παράλειψη).
        """
        E.get(endpoint.path)                     # φράγμα: μόνο επαληθευμένα
        params = {k: v for k, v in params.items() if v is not None and v != ""}
        missing = [p for p in endpoint.required if p not in params]
        if missing:
            raise ValueError(
                f"{endpoint.path}: λείπουν υποχρεωτικές παράμετροι {missing}")
        unknown = set(params) - set(endpoint.required) - set(endpoint.optional)
        if unknown:
            raise ValueError(
                f"{endpoint.path}: άγνωστες παράμετροι {sorted(unknown)} — "
                f"επιτρεπτές: {sorted(set(endpoint.required) | set(endpoint.optional))}")

        if use_cache and self.cache is not None:
            hit = self.cache.get(endpoint.path, params)
            if hit is not None:
                payload, _ = hit
                self.stats.record(endpoint.path, True, 0.0, 0)
                log.debug("cache hit %s", endpoint.path)
                return payload

        self._check_budget(endpoint.cost_units)

        last_exc: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries):
            self._bucket.acquire()
            t0 = time.monotonic()
            try:
                if self._client is not None:
                    payload = self._request_httpx(endpoint.path, params)
                else:
                    payload = self._request_curl(endpoint.path, params)
                dt = (time.monotonic() - t0) * 1000
                self.stats.record(endpoint.path, False, dt, endpoint.cost_units)
                if use_cache and self.cache is not None:
                    self.cache.put(endpoint.path, params, payload,
                                   self._ttl_seconds(endpoint.ttl_family))
                log.debug("%s ok σε %.0fms", endpoint.path, dt)
                return payload

            except (HikerAuthError, HikerNotFound):
                self.stats.record(endpoint.path, False,
                                  (time.monotonic() - t0) * 1000,
                                  endpoint.cost_units, error=True)
                raise

            except Exception as exc:                     # noqa: BLE001
                last_exc = exc
                dt = (time.monotonic() - t0) * 1000
                retryable = getattr(exc, "retryable", True)
                is_last = attempt == self.cfg.max_retries - 1
                self.stats.record(endpoint.path, False, dt,
                                  endpoint.cost_units if attempt == 0 else 0,
                                  error=is_last)
                if not retryable or is_last:
                    break
                self.stats.retries += 1
                # Στη 2η αποτυχία δοκιμάζουμε τον server χωρίς Cloudflare.
                if attempt == 1 and self.cfg.fallback_url and \
                        self._base != self.cfg.fallback_url:
                    log.warning("Εναλλαγή σε fallback server HikerAPI")
                    self._base = self.cfg.fallback_url
                backoff = min(20.0, (2 ** attempt) * 1.5) + random.uniform(0, 0.8)
                log.warning("%s απέτυχε (%s) — retry σε %.1fs",
                            endpoint.path, type(exc).__name__, backoff)
                time.sleep(backoff)

        # Τελευταία γραμμή άμυνας: ληγμένα δεδομένα αντί για ολική αποτυχία.
        if self.cache is not None:
            hit = self.cache.get(endpoint.path, params, allow_stale=True)
            if hit is not None:
                payload, _ = hit
                self.stats.stale_hits += 1
                log.warning("Χρήση ΛΗΓΜΕΝΩΝ δεδομένων cache για %s", endpoint.path)
                return payload
        raise last_exc if last_exc else HikerError("Άγνωστη αποτυχία", 0, endpoint.path)

    def try_call(self, endpoint: E.Endpoint, **params: Any) -> Optional[Any]:
        """Ήπια εκδοχή: επιστρέφει None αντί να ρίξει, εκτός από auth/budget."""
        try:
            return self.call(endpoint, **params)
        except (HikerAuthError, BudgetExceeded):
            raise
        except Exception as exc:                          # noqa: BLE001
            log.info("Παράλειψη %s: %s", endpoint.path, type(exc).__name__)
            return None

    def balance(self) -> Optional[dict]:
        """Υπόλοιπο credits. Δεν χρεώνεται και δεν μπαίνει σε cache."""
        try:
            return self.call(E.BALANCE, use_cache=False)
        except Exception as exc:                          # noqa: BLE001
            log.warning("Αδύνατη ανάγνωση υπολοίπου: %s", type(exc).__name__)
            return None

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

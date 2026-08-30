"""
Επίπεδο μοντέλου με ΕΓΓΥΗΜΕΝΑ δομημένη έξοδο.

Γιατί δεν ζητάμε απλώς «γύρνα JSON»: τα μοντέλα κατά καιρούς τυλίγουν το
JSON σε markdown, προσθέτουν πρόλογο, ή παραλείπουν πεδία. Σε production
pipeline αυτό σημαίνει τυχαίες αποτυχίες.

Λύση: `output_config.format` με JSON Schema. Το API επιβάλλει τη δομή στην
έξοδο, οπότε το πρώτο text block είναι πάντα έγκυρο JSON του σχήματός μας.

ΠΡΟΣΟΧΗ ΣΤΙΣ ΠΑΡΑΜΕΤΡΟΥΣ ΑΝΑ ΜΟΝΤΕΛΟ — τα σφάλματα εδώ είναι 400, όχι
υποβάθμιση:

  • `thinking={"type": "enabled", "budget_tokens": N}` ΑΠΟΡΡΙΠΤΕΤΑΙ στα
    Opus 5 / 4.8 / 4.7, Sonnet 5, Fable 5. Η σωστή μορφή είναι
    `{"type": "adaptive"}` και το βάθος ελέγχεται με `output_config.effort`.
  • `temperature` / `top_p` / `top_k` ΑΠΟΡΡΙΠΤΟΝΤΑΙ στα ίδια μοντέλα.

Γι' αυτό οι παράμετροι επιλέγονται από τις ΔΥΝΑΤΟΤΗΤΕΣ του μοντέλου, όχι
στατικά — αλλιώς μια αλλαγή μοντέλου στο .env σπάει όλο το σύστημα.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from typing import Any, Optional

from ...config import ModelConfig
from ...errors import LLMError, SchemaViolation
from ...logging_setup import get_logger

log = get_logger("llm")

# Οικογένειες μοντέλων με adaptive thinking και ΧΩΡΙΣ sampling/budget_tokens.
_MODERN = re.compile(
    r"claude-(fable-5|mythos-5|opus-5|opus-4-[678]|sonnet-5|sonnet-4-6)")

# Χαρτογράφηση «βάθους σκέψης» σε επίπεδο effort. Κρατάμε τη σημασιολογία
# των παλιών κλήσεων (budget σε tokens) χωρίς να στέλνουμε την απαγορευμένη
# παράμετρο.
def _effort_for(depth: int) -> str:
    if depth <= 0:
        return "low"
    if depth < 3000:
        return "medium"
    if depth < 6000:
        return "high"
    return "xhigh"


def supports_adaptive_thinking(model: str) -> bool:
    return bool(_MODERN.search(model or ""))


def supports_sampling(model: str) -> bool:
    """Τα σύγχρονα μοντέλα απορρίπτουν temperature/top_p/top_k με 400."""
    return not supports_adaptive_thinking(model or "")


# Λέξεις-κλειδιά επικύρωσης που δεν είναι εγγυημένα αποδεκτές από το
# `output_config.format`. Δεν τις πετάμε — τις μετατρέπουμε σε οδηγία μέσα
# στο `description`, ώστε το μοντέλο να τις τηρεί χωρίς κίνδυνο 400.
_RISKY_KEYWORDS = {
    "minItems": "τουλάχιστον {} στοιχεία",
    "maxItems": "το πολύ {} στοιχεία",
    "minimum": "ελάχιστη τιμή {}",
    "maximum": "μέγιστη τιμή {}",
    "minLength": "τουλάχιστον {} χαρακτήρες",
    "maxLength": "το πολύ {} χαρακτήρες",
    "pattern": "μοτίβο {}",
    "format": "μορφή {}",
}


def strictify(schema: dict) -> dict:
    """
    Κάνει ένα JSON Schema συμβατό με `output_config.format`.

    Τρεις μετασχηματισμοί, ο καθένας για συγκεκριμένο τρόπο αποτυχίας:

    1. `additionalProperties: false` σε κάθε object — απαίτηση του strict mode.
    2. `required` με ΟΛΑ τα properties. Τα «προαιρετικά» πεδία δεν χάνονται:
       γίνονται υποχρεωτικά και το μοντέλο επιστρέφει κενή τιμή όταν δεν έχει
       τι να βάλει — ασφαλέστερο για τον downstream κώδικα από απόν κλειδί.
    3. Λέξεις-κλειδιά επικύρωσης που ίσως δεν υποστηρίζονται (`minItems`,
       `maximum`, …) μεταφέρονται στο `description`. Έτσι η πρόθεση φτάνει
       στο μοντέλο ως οδηγία αντί να ρισκάρουμε απόρριψη ολόκληρου του
       αιτήματος με 400 — που θα έριχνε όλο το pipeline.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)

    hints = []
    for key, template in _RISKY_KEYWORDS.items():
        if key in out:
            hints.append(template.format(out.pop(key)))
    if hints:
        desc = out.get("description", "")
        out["description"] = (desc + " " if desc else "") + "(" + ", ".join(hints) + ")"

    if out.get("type") == "object":
        props = {k: strictify(v) for k, v in (out.get("properties") or {}).items()}
        out["properties"] = props
        out["required"] = list(props)
        out["additionalProperties"] = False
    elif out.get("type") == "array" and "items" in out:
        out["items"] = strictify(out["items"])
    return out


class AnthropicClient:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        try:
            import anthropic                       # noqa: PLC0415
        except ImportError as exc:                 # pragma: no cover
            raise LLMError(
                "Λείπει το πακέτο anthropic. Εγκατάσταση:\n"
                "    python3 -m pip install --user anthropic") from exc
        self._sdk = anthropic
        self._lock = threading.Lock()
        self._client = self._build_client()
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def _build_client(self):
        """
        Πελάτης με ΛΕΠΤΟΜΕΡΗ timeouts.

        Ένα ενιαίο timeout 300s ήταν λάθος: σε streaming, το «read» ισχύει ανά
        κομμάτι δεδομένων. Μια νεκρή σύνδεση περίμενε 5 ολόκληρα λεπτά πριν
        αποτύχει, και με 5 επαναλήψεις μια ανάλυση κόλλησε για ~60 λεπτά.

        Με χωριστά όρια, μια ροή που δεν στέλνει τίποτα για 90 δευτερόλεπτα
        θεωρείται νεκρή αμέσως. Το `keepalive_expiry` πετά αδρανείς συνδέσεις
        ώστε μια μακρόβια διεργασία server να μη σερβίρει σάπιο socket.
        """
        try:
            import httpx2 as _httpx                # anthropic 1.x
        except ImportError:
            import httpx as _httpx                 # anthropic 0.x
        return self._sdk.Anthropic(
            api_key=self.cfg.anthropic_key or None,
            max_retries=0,                          # οι επαναλήψεις είναι δικές μας
            timeout=_httpx.Timeout(connect=20.0, read=90.0, write=60.0, pool=20.0),
        )

    def _reset_client(self) -> None:
        """Ξαναχτίζει τον πελάτη — η μόνη θεραπεία για σπασμένο pool συνδέσεων."""
        with self._lock:
            try:
                self._client.close()
            except Exception:                       # noqa: BLE001
                pass
            self._client = self._build_client()
        log.info("Ο πελάτης Anthropic ξαναχτίστηκε (νέες συνδέσεις)")

    # ── εσωτερικά ────────────────────────────────────────────────────
    def _track(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["calls"] += 1

    @staticmethod
    def _describe(exc: Exception) -> str:
        """
        Αναγνώσιμη περιγραφή σφάλματος API.

        Το σκέτο «APIStatusError» δεν λέει τίποτα σε κανέναν: μπορεί να είναι
        υπερφόρτωση, rate limit, πολύ μεγάλο αίτημα ή εξαντλημένο υπόλοιπο —
        και η ενέργεια του χρήστη διαφέρει ριζικά σε κάθε περίπτωση.
        """
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        body = ""
        for attr in ("message", "body"):
            v = getattr(exc, attr, None)
            if v:
                body = str(v)[:300]
                break
        if not body:
            body = str(exc)[:300]
        hint = {
            429: "Rate limit — το σύστημα θα ξαναπροσπαθήσει. Αν επιμένει, "
                 "μείωσε τις παράλληλες αναλύσεις.",
            529: "Το API είναι υπερφορτωμένο αυτή τη στιγμή. Δοκίμασε ξανά σε λίγο.",
            500: "Προσωρινό σφάλμα διακομιστή.",
            502: "Προσωρινό σφάλμα δικτύου προς το API.",
            503: "Η υπηρεσία είναι προσωρινά μη διαθέσιμη.",
            400: "Το αίτημα απορρίφθηκε — πιθανώς πολύ μεγάλο ή λάθος παράμετρος.",
            401: "Άκυρο ANTHROPIC_API_KEY.",
            403: "Το κλειδί δεν έχει δικαίωμα για αυτό το μοντέλο.",
            413: "Το αίτημα είναι πολύ μεγάλο — μείωσε το VRGR_MAX_FRAMES.",
        }.get(status, "")
        parts = [name]
        if status:
            parts.append(f"HTTP {status}")
        if hint:
            parts.append(hint)
        if body:
            parts.append(f"[{body}]")
        return " · ".join(parts)

    def _with_retry(self, fn, attempts: int = 4, what: str = "κλήση"):
        """
        Υπομονετικό backoff.

        Μετρήθηκε: η κλήση όρασης απέτυχε με παροδικό APIStatusError και οι
        τρεις γρήγορες επαναλήψεις (2,8s / 4,9s) εξαντλήθηκαν πριν περάσει η
        υπερφόρτωση — μια ανάλυση 5 λεπτών χάθηκε για ένα σφάλμα λίγων
        δευτερολέπτων. Πέντε προσπάθειες με κλιμάκωση έως 45s κοστίζουν στη
        χειρότερη περίπτωση ~2 λεπτά αναμονής και σώζουν την εκτέλεση.
        """
        last: Optional[Exception] = None
        for i in range(attempts):
            try:
                return fn()
            except Exception as exc:               # noqa: BLE001
                last = exc
                name = type(exc).__name__
                # Σφάλματα αιτήματος δεν διορθώνονται με επανάληψη.
                if name in ("BadRequestError", "AuthenticationError",
                            "PermissionDeniedError", "NotFoundError"):
                    raise LLMError(f"{what}: {self._describe(exc)}") from exc
                if i == attempts - 1:
                    break
                # Σφάλμα δικτύου = πιθανώς σπασμένη σύνδεση στο pool. Το retry
                # πάνω στην ίδια σύνδεση απλώς ξαναποτυγχάνει· χτίζουμε νέα.
                if name in ("APIConnectionError", "APITimeoutError",
                            "ReadTimeout", "ConnectError", "RemoteProtocolError"):
                    self._reset_client()
                wait = min(45.0, (2 ** i) * 3.0) + random.uniform(0, 1.5)
                log.warning("%s απέτυχε (%s) — retry σε %.1fs",
                            what, self._describe(exc), wait)
                time.sleep(wait)
        raise LLMError(f"{what} απέτυχε μετά από {attempts} προσπάθειες: "
                       f"{self._describe(last) if last else 'άγνωστο'}") from last

    # ── δημόσιο API ──────────────────────────────────────────────────
    def structured(self, *, system: str, content: list, schema: dict,
                   tool_name: str = "submit", tool_description: str = "",
                   model: Optional[str] = None, max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None,
                   thinking_budget: int = 0) -> dict:
        """
        Επιστρέφει dict που ακολουθεί το `schema`.

        Τα `tool_name` / `tool_description` κρατούνται για συμβατότητα με τους
        καλούντες και χρησιμοποιούνται μόνο για μηνύματα σφάλματος.
        """
        model = model or self.cfg.writer_model
        modern = supports_adaptive_thinking(model)

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens or self.cfg.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {
                "format": {"type": "json_schema", "schema": strictify(schema)},
            },
        }
        if modern:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"]["effort"] = _effort_for(thinking_budget)
        elif temperature is not None:
            kwargs["temperature"] = temperature

        # STREAMING ΥΠΟΧΡΕΩΤΙΚΑ.
        #
        # Μετρήθηκε ζωντανά: μη-streaming κλήση με adaptive thinking και
        # max_tokens 6-8K κόλλησε >10 λεπτά και χτύπησε το HTTP timeout του
        # SDK — μετά ακολούθησαν retries, δηλαδή έως 30 λεπτά για ΜΙΑ κλήση.
        # Το streaming κρατά τη σύνδεση ζωντανή· το `get_final_message()`
        # επιστρέφει το ίδιο αντικείμενο απόκρισης.
        def _call():
            with self._client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()

        response = self._with_retry(_call, what=f"structured({tool_name})")
        self._track(response)

        if getattr(response, "stop_reason", "") == "refusal":
            detail = getattr(response, "stop_details", None)
            raise LLMError(
                f"Το μοντέλο αρνήθηκε το αίτημα «{tool_name}»"
                + (f" ({getattr(detail, 'category', '')})" if detail else ""))

        text = "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text").strip()
        if not text:
            raise LLMError(
                f"Κενή απόκριση για «{tool_name}» — blocks: "
                f"{[getattr(b, 'type', '?') for b in response.content]}")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaViolation(
                f"Μη έγκυρο JSON για «{tool_name}»: {text[:200]}") from exc
        if not isinstance(data, dict):
            raise SchemaViolation(
                f"Αναμενόταν αντικείμενο για «{tool_name}», ελήφθη {type(data).__name__}")
        return data

    def text(self, *, system: str, content: list,
             model: Optional[str] = None, max_tokens: int = 2000,
             temperature: Optional[float] = None) -> str:
        model = model or self.cfg.fast_model
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        if temperature is not None and supports_sampling(model):
            kwargs["temperature"] = temperature

        def _call():
            with self._client.messages.stream(**kwargs) as stream:
                return stream.get_final_message()

        response = self._with_retry(_call, what="text")
        self._track(response)
        return "".join(b.text for b in response.content
                       if getattr(b, "type", "") == "text").strip()

    def usage_summary(self) -> dict:
        return dict(self.usage)


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def json_block(label: str, data: Any, limit: int = 30000) -> dict:
    """Δεδομένα ως κείμενο για το prompt, με κόψιμο ασφαλείας."""
    blob = json.dumps(data, ensure_ascii=False, indent=1, default=str)
    if len(blob) > limit:
        blob = blob[:limit] + "\n… (κόπηκε)"
    return {"type": "text", "text": f"<{label}>\n{blob}\n</{label}>"}

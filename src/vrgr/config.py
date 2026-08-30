"""
Ρυθμίσεις — αποκλειστικά από environment / .env.

ΚΑΝΕΝΑ secret δεν γράφεται ποτέ σε πηγαίο κώδικα ή σε config αρχείο του repo.
Το .env είναι gitignored· το .env.example περιέχει μόνο ονόματα μεταβλητών.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .errors import ConfigError

_TRUE = {"1", "true", "yes", "on", "ναι"}


def _load_dotenv(path: Path) -> None:
    """Ελάχιστος parser .env — χωρίς εξάρτηση σε python-dotenv.

    Το περιβάλλον έχει πάντα προτεραιότητα: μια ήδη ορισμένη μεταβλητή
    δεν αντικαθίσταται από το αρχείο.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_streamlit_secrets() -> None:
    """
    Γέφυρα προς τα `st.secrets` για ανάπτυξη σε Streamlit Cloud.

    Στο cloud δεν υπάρχει αρχείο `.env` — τα κλειδιά δίνονται από το panel του
    Streamlit. Τα αντιγράφουμε στο περιβάλλον ΜΟΝΟ όσα λείπουν, ώστε η
    υπόλοιπη εφαρμογή να μη χρειάζεται να ξέρει πού τρέχει.
    """
    try:
        import streamlit as st                      # noqa: PLC0415
        secrets = st.secrets
    except Exception:                               # noqa: BLE001
        return                                      # δεν τρέχουμε σε Streamlit
    for key in ("HIKER_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY",
                "OPENAI_API_KEY", "GROQ_API_KEY", "VRGR_DATA_DIR",
                "VRGR_EMBEDDING_PROVIDER", "VRGR_ASR_PROVIDER",
                "VRGR_VISION_MODEL", "VRGR_WRITER_MODEL", "VRGR_FAST_MODEL",
                "HIKER_BUDGET_PER_RUN", "VRGR_MAX_FRAMES"):
        if os.environ.get(key):
            continue
        try:
            value = secrets[key]
        except Exception:                           # noqa: BLE001
            continue
        if value:
            os.environ[key] = str(value)


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key) or default)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key) or default)
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key)
    return v.lower() in _TRUE if v else default


@dataclass(frozen=True)
class HikerConfig:
    api_key: str
    base_url: str = "https://api.hikerapi.com"
    fallback_url: str = "https://api.instagrapi.com"
    timeout_s: int = 60
    max_retries: int = 3
    rate_limit_rps: float = 3.0
    budget_per_run: int = 120
    ttl_profile_min: int = 720
    ttl_media_min: int = 1440
    ttl_search_min: int = 180
    ttl_hashtag_min: int = 360
    # Το HikerAPI απορρίπτει με 403 το default python-httpx User-Agent.
    # Επιβεβαιωμένο εμπειρικά· γι' αυτό στέλνουμε ρητό UA.
    user_agent: str = "curl/8.7.1"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class ModelConfig:
    anthropic_key: str
    vision_model: str = "claude-opus-5"
    writer_model: str = "claude-opus-5"
    fast_model: str = "claude-haiku-4-5"
    max_tokens: int = 8000
    temperature: float = 1.0

    @property
    def enabled(self) -> bool:
        return bool(self.anthropic_key)


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "local"          # local | voyage | openai
    model: str = "voyage-3.5"
    voyage_key: str = ""
    openai_key: str = ""
    dim: int = 512                   # διάσταση για τον local provider


@dataclass(frozen=True)
class VideoConfig:
    max_frames: int = 14
    hook_window_s: float = 3.0
    frame_width: int = 768
    scene_threshold: float = 0.30
    asr_provider: str = "none"       # none | openai | groq
    groq_key: str = ""
    openai_key: str = ""


@dataclass(frozen=True)
class Settings:
    hiker: HikerConfig
    models: ModelConfig
    embeddings: EmbeddingConfig
    video: VideoConfig
    data_dir: Path
    db_backend: str = "sqlite"
    postgres_dsn: str = ""
    market: str = "GR"
    language: str = "el"
    log_level: str = "INFO"
    log_format: str = "text"
    config_dir: Path = field(default_factory=lambda: Path("config"))

    # ---- παράγωγες διαδρομές -------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "vrgr.db"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "hiker_cache.db"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.media_dir, self.runs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def require_hiker(self) -> HikerConfig:
        if not self.hiker.enabled:
            raise ConfigError(
                "Λείπει το HIKER_API_KEY. Βάλ' το στο .env "
                "(δες .env.example). Χωρίς αυτό δεν γίνεται έρευνα πραγματικών δεδομένων."
            )
        return self.hiker

    def require_models(self) -> ModelConfig:
        if not self.models.enabled:
            raise ConfigError(
                "Λείπει το ANTHROPIC_API_KEY. Βάλ' το στο .env (δες .env.example)."
            )
        return self.models


_cached: Optional[Settings] = None


def load_settings(root: Optional[Path] = None, reload: bool = False) -> Settings:
    """Φορτώνει τις ρυθμίσεις μία φορά ανά διεργασία."""
    global _cached
    if _cached is not None and not reload:
        return _cached

    root = root or Path(__file__).resolve().parents[2]
    _load_dotenv(root / ".env")
    _load_streamlit_secrets()

    data_dir = Path(_env("VRGR_DATA_DIR", str(root / "data"))).expanduser()
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()

    s = Settings(
        hiker=HikerConfig(
            api_key=_env("HIKER_API_KEY"),
            base_url=_env("HIKER_BASE_URL", "https://api.hikerapi.com"),
            fallback_url=_env("HIKER_FALLBACK_URL", "https://api.instagrapi.com"),
            timeout_s=_env_int("HIKER_TIMEOUT_S", 60),
            max_retries=_env_int("HIKER_MAX_RETRIES", 3),
            rate_limit_rps=_env_float("HIKER_RATE_LIMIT_RPS", 3.0),
            budget_per_run=_env_int("HIKER_BUDGET_PER_RUN", 120),
            ttl_profile_min=_env_int("HIKER_CACHE_TTL_PROFILE_MIN", 720),
            ttl_media_min=_env_int("HIKER_CACHE_TTL_MEDIA_MIN", 1440),
            ttl_search_min=_env_int("HIKER_CACHE_TTL_SEARCH_MIN", 180),
            ttl_hashtag_min=_env_int("HIKER_CACHE_TTL_HASHTAG_MIN", 360),
        ),
        models=ModelConfig(
            anthropic_key=_env("ANTHROPIC_API_KEY"),
            vision_model=_env("VRGR_VISION_MODEL", "claude-opus-5"),
            writer_model=_env("VRGR_WRITER_MODEL", "claude-opus-5"),
            fast_model=_env("VRGR_FAST_MODEL", "claude-haiku-4-5"),
        ),
        embeddings=EmbeddingConfig(
            provider=_env("VRGR_EMBEDDING_PROVIDER", "local").lower(),
            model=_env("VRGR_EMBEDDING_MODEL", "voyage-3.5"),
            voyage_key=_env("VOYAGE_API_KEY"),
            openai_key=_env("OPENAI_API_KEY"),
        ),
        video=VideoConfig(
            max_frames=_env_int("VRGR_MAX_FRAMES", 14),
            hook_window_s=_env_float("VRGR_HOOK_WINDOW_S", 3.0),
            asr_provider=_env("VRGR_ASR_PROVIDER", "none").lower(),
            groq_key=_env("GROQ_API_KEY"),
            openai_key=_env("OPENAI_API_KEY"),
        ),
        data_dir=data_dir,
        db_backend=_env("VRGR_DB_BACKEND", "sqlite").lower(),
        postgres_dsn=_env("VRGR_POSTGRES_DSN"),
        market=_env("VRGR_MARKET", "GR"),
        language=_env("VRGR_LANGUAGE", "el"),
        log_level=_env("VRGR_LOG_LEVEL", "INFO").upper(),
        log_format=_env("VRGR_LOG_FORMAT", "text").lower(),
        config_dir=root / "config",
    )
    s.ensure_dirs()
    _cached = s
    return s


def secret_values() -> list:
    """Όλες οι τιμές που δεν επιτρέπεται να εμφανιστούν σε log/έξοδο."""
    keys = (
        "HIKER_API_KEY", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY",
        "OPENAI_API_KEY", "GROQ_API_KEY",
    )
    return [v for v in (os.environ.get(k, "") for k in keys) if len(v) >= 8]

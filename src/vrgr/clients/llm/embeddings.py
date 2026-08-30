"""
Διανυσματικές αναπαραστάσεις για σημασιολογική αναζήτηση στη μνήμη.

Γιατί υπάρχει `local` provider και γιατί είναι το default:

  • Η Anthropic δεν παρέχει embeddings API.
  • Εξωτερικός πάροχος (Voyage/OpenAI) σημαίνει ακόμη ένα κλειδί, ακόμη ένα
    κόστος, ακόμη ένα σημείο αποτυχίας — για λειτουργία που πρέπει να
    δουλεύει από την πρώτη μέρα.
  • Για ΜΙΚΡΑ ΕΛΛΗΝΙΚΑ ΚΕΙΜΕΝΑ (λεζάντες 10-40 λέξεων) ένας hashing
    vectorizer πάνω σε χαρακτηρο-n-grams είναι εκπληκτικά ανταγωνιστικός:
    τα n-grams απορροφούν την ελληνική κλίση («καφές/καφέδες/καφέ»), κάτι
    που τα word embeddings συχνά χάνουν σε μορφολογικά πλούσιες γλώσσες.

Για production με δεκάδες χιλιάδες posts, το `voyage` (πολύγλωσσο) δίνει
καλύτερη σημασιολογία. Η εναλλαγή είναι μία μεταβλητή περιβάλλοντος και
δεν αγγίζει τον υπόλοιπο κώδικα.
"""
from __future__ import annotations

import hashlib
import struct
import numpy as np

from ... import greek as G
from ...config import EmbeddingConfig
from ...logging_setup import get_logger

log = get_logger("embeddings")


def _h(token: str, salt: str, dim: int) -> int:
    digest = hashlib.blake2b(f"{salt}\x00{token}".encode("utf-8"),
                             digest_size=8).digest()
    return struct.unpack("<Q", digest)[0] % dim


def _sign(token: str) -> float:
    """Signed hashing — μειώνει το bias από τις συγκρούσεις κατακερματισμού."""
    return 1.0 if hashlib.blake2b(token.encode("utf-8"),
                                  digest_size=2).digest()[0] & 1 else -1.0


class LocalEmbedder:
    """Hashing vectorizer: λέξεις + χαρακτηρο-n-grams, sublinear tf, L2."""

    name = "local-hash-v1"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = out[i]
            words = G.tokenize(text, drop_stopwords=True)
            for w in words:
                vec[_h(w, "w", self.dim)] += 1.6 * _sign(w)
            for a, b in zip(words, words[1:]):                # bigrams λέξεων
                bg = f"{a}_{b}"
                vec[_h(bg, "b", self.dim)] += 1.1 * _sign(bg)
            for n in (3, 4, 5):                               # χαρακτηρο-n-grams
                for gram in G.char_ngrams(text, n):
                    if gram.strip():
                        vec[_h(gram, f"c{n}", self.dim)] += 0.55 * _sign(gram)
            for tag in G.extract_hashtags(text):              # tags βαραίνουν
                vec[_h(tag, "t", self.dim)] += 2.2 * _sign(tag)
            # Sublinear scaling: μία λέξη 20 φορές δεν είναι 20x πιο σημαντική.
            nz = vec != 0
            vec[nz] = np.sign(vec[nz]) * (1.0 + np.log1p(np.abs(vec[nz])))
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
        return out


class VoyageEmbedder:
    """Πολύγλωσσος πάροχος — συνιστάται για production όγκο."""

    def __init__(self, api_key: str, model: str = "voyage-3.5"):
        import voyageai                       # noqa: PLC0415
        self._client = voyageai.Client(api_key=api_key)
        self.model = model
        self.name = f"voyage:{model}"
        self.dim = 1024

    def embed(self, texts: list) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 96):
            batch = [t[:8000] for t in texts[i:i + 96]]
            res = self._client.embed(batch, model=self.model, input_type="document")
            vectors.extend(res.embeddings)
        arr = np.asarray(vectors, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return arr


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai                          # noqa: PLC0415
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai:{model}"
        self.dim = 1536

    def embed(self, texts: list) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), 128):
            batch = [t[:8000] for t in texts[i:i + 128]]
            res = self._client.embeddings.create(model=self.model, input=batch)
            vectors.extend(d.embedding for d in res.data)
        arr = np.asarray(vectors, dtype=np.float32)
        self.dim = arr.shape[1] if arr.size else self.dim
        return arr


def build_embedder(cfg: EmbeddingConfig):
    """Επιλογή παρόχου με σιωπηλή υποχώρηση στον local αν λείπει κλειδί/πακέτο."""
    provider = (cfg.provider or "local").lower()
    if provider == "voyage" and cfg.voyage_key:
        try:
            return VoyageEmbedder(cfg.voyage_key, cfg.model)
        except Exception as exc:                       # noqa: BLE001
            log.warning("Voyage μη διαθέσιμο (%s) — χρήση local embedder",
                        type(exc).__name__)
    elif provider == "openai" and cfg.openai_key:
        try:
            return OpenAIEmbedder(cfg.openai_key)
        except Exception as exc:                       # noqa: BLE001
            log.warning("OpenAI μη διαθέσιμο (%s) — χρήση local embedder",
                        type(exc).__name__)
    return LocalEmbedder(cfg.dim)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ομοιότητα ενός διανύσματος προς πίνακα διανυσμάτων."""
    if b.ndim == 1:
        b = b.reshape(1, -1)
    an = a / (np.linalg.norm(a) or 1.0)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return bn @ an


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=np.float32)
    return arr[:dim] if arr.size >= dim else np.pad(arr, (0, dim - arr.size))


def jaccard_ngram(a: str, b: str, n: int = 4) -> float:
    """Ομοιότητα για anti-plagiarism — ανεξάρτητη από τον embedder."""
    sa, sb = set(G.char_ngrams(a, n)), set(G.char_ngrams(b, n))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

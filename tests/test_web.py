"""Web layer: streaming ανέβασμα, συντήρηση δίσκου, φίλτρα hashtag."""
import io
import os
import time

import pytest

from vrgr.generation.hashtags import (CROSS_PLATFORM, EMPTY_SIGNALS,
                                      build_candidates, is_branded,
                                      is_cross_platform)
from vrgr.schemas import MinedPatterns, VideoContent
from vrgr.web.server import MultipartError, stream_multipart


def _body(boundary: bytes, blob: bytes, fields: dict) -> bytes:
    parts = []
    for k, v in fields.items():
        parts.append(b"--" + boundary
                     + f'\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                     .encode("utf-8"))
    parts.append(b"--" + boundary
                 + b'\r\nContent-Disposition: form-data; name="video"; filename="reel.mp4"\r\n'
                 + b"Content-Type: video/mp4\r\n\r\n" + blob + b"\r\n")
    parts.append(b"--" + boundary + b"--\r\n")
    return b"".join(parts)


def test_upload_streams_to_disk_without_loading_it_all(tmp_path):
    """
    Ο λόγος ύπαρξης του streaming parser.

    Ένα Reel 4K από κινητό είναι συχνά 150-300 MB. Με `rfile.read(length)` η
    κορύφωση μνήμης έφτανε ~1,2 GB μετά τα αντίγραφα — αρκετό για να σκοτώσει
    τη διεργασία.
    """
    boundary = b"----VRGRtest"
    blob = os.urandom(3_000_000)
    body = _body(boundary, blob, {"context": "ελληνικό κείμενο", "captions": "8"})

    fields, files = stream_multipart(io.BytesIO(body), len(body), boundary,
                                     tmp_path, 400 * 1024 * 1024)
    assert fields["context"] == "ελληνικό κείμενο"
    assert fields["captions"] == "8"
    filename, path, size = files["video"]
    assert filename == "reel.mp4"
    assert size == len(blob)
    assert path.read_bytes() == blob


def test_upload_handles_boundary_split_across_chunks(tmp_path):
    """Το boundary μπορεί να πέσει ακριβώς πάνω σε όριο κομματιού των 256 KB."""
    boundary = b"----VRGRtest"
    from vrgr.web.server import CHUNK
    # Μέγεθος επιλεγμένο ώστε το τερματικό boundary να κόβεται στα δύο.
    blob = os.urandom(CHUNK * 2 - 7)
    body = _body(boundary, blob, {"context": "x"})
    _, files = stream_multipart(io.BytesIO(body), len(body), boundary,
                                tmp_path, 400 * 1024 * 1024)
    assert files["video"][2] == len(blob)
    assert files["video"][1].read_bytes() == blob


def test_upload_rejects_oversized_request(tmp_path):
    with pytest.raises(MultipartError):
        stream_multipart(io.BytesIO(b""), 500 * 1024 * 1024, b"B",
                         tmp_path, 400 * 1024 * 1024)


def test_maintenance_removes_old_but_keeps_knowledge(tmp_path, monkeypatch):
    from vrgr.config import (EmbeddingConfig, HikerConfig, ModelConfig,
                             Settings, VideoConfig)
    from vrgr.maintenance import cleanup, usage
    s = Settings(hiker=HikerConfig(api_key=""), models=ModelConfig(anthropic_key=""),
                 embeddings=EmbeddingConfig(), video=VideoConfig(),
                 data_dir=tmp_path / "data")
    s.ensure_dirs()
    s.db_path.write_bytes(os.urandom(50_000))          # «βάση γνώσης»

    old = time.time() - 10 * 86400
    frames = s.media_dir / "oldrun" / "frames"
    frames.mkdir(parents=True)
    (frames / "a.jpg").write_bytes(os.urandom(40_000))
    os.utime(frames.parent, (old, old))
    fresh = s.media_dir / "newrun"
    fresh.mkdir()
    (fresh / "b.jpg").write_bytes(os.urandom(40_000))

    up = s.data_dir / "uploads"
    up.mkdir(exist_ok=True)
    orphan = up / "orphan.mp4"
    orphan.write_bytes(os.urandom(20_000))
    os.utime(orphan, (old, old))

    db_before = s.db_path.stat().st_size
    freed = cleanup(s)

    assert freed["frames_dirs"] == 1 and freed["uploads"] == 1
    assert not (s.media_dir / "oldrun").exists()
    assert fresh.exists()                              # πρόσφατα δεν πειράζονται
    assert s.db_path.stat().st_size == db_before       # η γνώση ΔΕΝ αγγίζεται
    assert usage(s)["βάση_γνώσης"] == db_before


def test_branded_hashtags_are_rejected():
    """
    Το προσωπικό brand hashtag άλλου creator δεν φέρνει ποτέ κοινό σε εσένα.

    Μετρήθηκε σε πραγματική εκτέλεση: η ανάλυση ανύψωσης πρότεινε
    «#chryssanthemis» επειδή ο ίδιος creator κυριαρχούσε στα outliers.
    """
    creators = {"chryssanthemis_official", "kate_pavli", "asteroeidis.aigio"}
    assert is_branded("chryssanthemis", creators)
    assert is_branded("katepavli", creators)
    assert not is_branded("greekmemes", creators)
    assert not is_branded("σχεσεις", creators)


def test_platform_tags_rejected_even_with_greek_suffix():
    """
    Η ακριβής αντιστοίχιση δεν αρκεί.

    Σε πραγματική εκτέλεση πέρασε το «#youtubeshortsελληνικα» — εξίσου
    άχρηστο σε Instagram Reel με το σκέτο «#youtubeshorts».
    """
    for t in ("youtubeshortsελληνικα", "tiktokviralgreece", "capcutedit",
              "viraltiktok", "youtubeshorts"):
        assert is_cross_platform(t), t
    for t in ("ελληνικοχιουμορ", "greekreels", "reelsgreece", "storytime"):
        assert not is_cross_platform(t), t


def test_candidate_filters_drop_useless_tags():
    content = VideoContent(niche="χιούμορ", sub_niche="σχέσεις")
    mined = MinedPatterns(outlier_sample_size=10, top_hashtags=[
        ("chryssanthemis", 4.1, 6), ("viraltiktok", 3.0, 5), ("foryou", 2.8, 4),
        ("youtubeshortsελληνικα", 2.6, 4),
        ("ελληνικοχιουμορ", 2.4, 7), ("σχεσεις", 1.9, 5)])
    cands = build_candidates(content, None, mined, {}, niche="χιούμορ",
                             creator_usernames={"chryssanthemis_official"})
    tags = {c.tag for c in cands}
    assert "chryssanthemis" not in tags            # brand άλλου
    assert not any(is_cross_platform(t) for t in tags)   # άλλη πλατφόρμα
    assert not (tags & EMPTY_SIGNALS)              # κενό σήμα
    assert "ελληνικοχιουμορ" in tags               # το χρήσιμο επιβίωσε


# ── επιβίωση μνήμης σε εφήμερο filesystem ─────────────────────────────

def test_memory_export_import_roundtrip(tmp_path):
    """
    Σε εφήμερο δίσκο (Streamlit Cloud) αυτό ΔΕΝ είναι πολυτέλεια — είναι ο
    μόνος τρόπος να επιβιώσει ό,τι έμαθε το σύστημα.
    """
    import sqlite3
    from vrgr.memory.db import Database
    from vrgr.seedstore import export_db, import_db

    src = tmp_path / "orig.db"
    db = Database(src)
    db.execute("INSERT INTO patterns (key, kind, niche, description_el, alpha, "
               "beta, last_seen, last_decayed) VALUES (?,?,?,?,?,?,?,?)",
               ("k1", "hashtag", "n", "δοκιμή", 5.0, 2.0, 0.0, 0.0))
    db.close()

    backup = export_db(src, tmp_path / "backup.db")
    assert backup.stat().st_size > 0

    target = tmp_path / "restored.db"
    target.write_bytes(b"")
    msg = import_db(backup.read_bytes(), target)
    assert "patterns" in msg
    conn = sqlite3.connect(str(target))
    assert conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0] == 1
    conn.close()


def test_memory_import_rejects_foreign_file(tmp_path):
    from vrgr.seedstore import import_db
    target = tmp_path / "db.sqlite"
    target.write_bytes(b"")
    with pytest.raises(Exception):
        import_db(b"this is not a database", target)


def test_seed_restore_only_when_database_missing(tmp_path, monkeypatch):
    """Δεν πατάει ποτέ υπάρχουσα βάση — αυτό θα έσβηνε πραγματική γνώση."""
    import vrgr.seedstore as S
    from vrgr.memory.db import Database

    seed = tmp_path / "seed.db"
    Database(seed).close()
    monkeypatch.setattr(S, "SEED_DB", seed)

    missing = tmp_path / "work.db"
    assert S.restore_if_missing(missing) is not None
    assert missing.exists()

    existing = tmp_path / "existing.db"
    payload = "ΥΠΑΡΧΟΥΣΑ ΓΝΩΣΗ".encode("utf-8")
    existing.write_bytes(payload)
    assert S.restore_if_missing(existing) is None
    assert existing.read_bytes() == payload

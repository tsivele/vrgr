"""Κοινά fixtures και ψεύτικες υπηρεσίες για τα tests."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vrgr.memory.db import Database                       # noqa: E402
from vrgr.schemas import ObservedPost, PostMetrics        # noqa: E402


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def config_dir():
    return Path(__file__).resolve().parents[1] / "config"


def make_post(media_id="1", username="user", followers=10_000, views=50_000,
              likes=2_000, comments=100, caption="Δοκιμαστική λεζάντα",
              hashtags=None, greek=1.0, days_ago=5, duration=15.0):
    """Δημιουργός δοκιμαστικών posts με ρεαλιστικά πεδία."""
    tags = hashtags if hashtags is not None else ["ελληνικοχιουμορ"]
    full = caption + " " + " ".join(f"#{t}" for t in tags)
    return ObservedPost(
        media_id=media_id, code=f"C{media_id}", username=username,
        creator_pk=username, followers_at_observation=followers,
        caption=full, caption_body=caption, hashtags=tags,
        greek_confidence=greek, duration_s=duration,
        taken_at=int(time.time()) - days_ago * 86400,
        metrics=PostMetrics(views=views, likes=likes, comments=comments))

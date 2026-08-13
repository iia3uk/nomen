"""Stability: deterministic seeds, atomic writes, fair tournaments, similarity."""

from __future__ import annotations

from pathlib import Path

from nomen.config import SimilarityConfig
from nomen.hashing import stable_seed
from nomen.models import Candidate
from nomen.persist import atomic_write_bytes
from nomen.scoring.overall import compute_overall
from nomen.scoring.scorer import score_candidate
from nomen.scoring.tournament import run_tournament
from nomen.similarity.engine import SimilarityEngine


def test_stable_seed_is_deterministic() -> None:
    a = stable_seed("test-seed-123")
    b = stable_seed("test-seed-123")
    assert a == b
    assert stable_seed("other") != a
    assert 0 <= a < 2**31


def test_atomic_write_replaces(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_bytes(path, b'{"ok": 1}')
    assert path.read_bytes() == b'{"ok": 1}'
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    atomic_write_bytes(path, b'{"ok": 2}')
    assert path.read_bytes() == b'{"ok": 2}'


def test_tournament_equalizes_match_counts() -> None:
    cands: list[Candidate] = []
    for i in range(16):
        c = Candidate(name=f"brand{i:02d}", generator="test")
        score_candidate(c)
        c.scores.beauty_score = 80
        c.scores.brand_score = 80
        c.scores.novelty_score = 80
        c.scores.collision_score = 80
        c.scores.overall = compute_overall(beauty=80, brand=80, novelty=80, collision=80)
        cands.append(c)
    result = run_tournament(cands, keep=4, n_users=20, seed=3)
    fights = [result.fought[c.name] for c in cands]
    assert min(fights) >= 10
    assert max(fights) - min(fights) <= 4


def test_similarity_catches_known_and_near_brands() -> None:
    sim = SimilarityEngine(SimilarityConfig())
    assert sim.reason("stripe")
    assert sim.reason("stripes")
    assert sim.reason("figmaa")
    assert not sim.reason("xqzztv")

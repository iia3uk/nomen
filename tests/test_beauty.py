"""BeautyScore: aesthetics-only model, synthetic rejects, multi-objective weights."""

from __future__ import annotations

from nomen.filters.offline import OfflineFilterBank
from nomen.models import Candidate
from nomen.scoring.beauty import BeautyModel, beauty_breakdown, beauty_score, get_beauty_model, passes_beauty_gates
from nomen.scoring.overall import W_BEAUTY, W_BRAND, W_COLLISION, W_NOVELTY, compute_overall
from nomen.scoring.scorer import score_candidate
from nomen.scoring.tournament import run_tournament


def test_weights_sum_and_beauty_dominates() -> None:
    assert abs(W_BEAUTY + W_BRAND + W_NOVELTY + W_COLLISION - 1.0) < 1e-9
    assert W_BEAUTY > W_BRAND > W_NOVELTY > W_COLLISION


def test_beauty_training_corpus_is_not_collapsed() -> None:
    brands = BeautyModel._load_brands()
    assert len(brands) > 500, f"beauty corpus collapsed to {len(brands)} names"
    get_beauty_model.cache_clear()
    assert get_beauty_model().n_brands > 500


def test_premium_brands_score_high() -> None:
    for name in ("figma", "stripe", "notion", "vercel", "linear", "raycast", "retool"):
        bd = beauty_breakdown(name)
        assert bd.beauty_score >= 70, f"{name} beauty={bd.beauty_score}"
        assert not bd.reject_reasons, f"{name} rejects={bd.reject_reasons}"


def test_synthetic_names_score_low() -> None:
    for name in ("zeflogho", "kadyogha", "plublumin", "gomaffive", "horrecte"):
        bd = beauty_breakdown(name)
        assert bd.beauty_score < 72 or bd.reject_reasons, (
            f"{name} unexpectedly pretty: {bd.beauty_score} rejects={bd.reject_reasons}"
        )
        ok, _ = passes_beauty_gates(name, min_beauty=72)
        assert not ok


def test_reserved_owned_brand_rejected() -> None:
    bank = OfflineFilterBank()
    reasons = bank.check("jasefly")
    assert any("reserved" in r for r in reasons)


def test_score_candidate_fills_beauty_and_collision() -> None:
    c = Candidate(name="velora", generator="test")
    score_candidate(c)
    assert c.scores.beauty_score > 0
    assert c.scores.collision_score > 0
    assert c.scores.brand_score > 0


def test_overall_not_novelty_dominated() -> None:
    # Extreme novelty cannot outrank strong beauty when others are equal-ish
    high_beauty = compute_overall(beauty=95, brand=85, novelty=40, collision=75)
    high_novelty = compute_overall(beauty=55, brand=85, novelty=99, collision=75)
    assert high_beauty > high_novelty
    # Beauty weight is the largest single term
    assert W_BEAUTY >= 0.35


def test_tournament_keeps_prettier_names() -> None:
    names = [
        ("figma", 92, 88, 60, 80),
        ("zeflogho", 35, 70, 90, 60),
        ("stripe", 91, 87, 55, 78),
        ("plublumin", 40, 72, 88, 62),
        ("vercel", 90, 86, 58, 77),
        ("gomaffive", 38, 68, 92, 58),
    ]
    cands: list[Candidate] = []
    for name, beauty, brand, nov, coll in names:
        c = Candidate(name=name, generator="test")
        score_candidate(c)
        c.scores.beauty_score = beauty
        c.scores.brand_score = brand
        c.scores.novelty_score = nov
        c.scores.collision_score = coll
        c.scores.overall = compute_overall(
            beauty=beauty, brand=brand, novelty=nov, collision=coll
        )
        cands.append(c)
    result = run_tournament(cands, keep=3, n_users=100, seed=7)
    winners = {c.name for c in result.winners}
    assert "figma" in winners or "stripe" in winners or "vercel" in winners
    assert "zeflogho" not in winners
    assert "gomaffive" not in winners


def test_beauty_model_does_not_require_verbatim_match() -> None:
    # Learned statistics should score a novel-but-natural name decently
    score = beauty_score("novira")
    assert 40 <= score <= 100


def test_beauty_gate_records_raw_score() -> None:
    bd = beauty_breakdown("zeflogho")
    assert bd.raw_beauty >= bd.beauty_score
    if bd.gate:
        assert bd.raw_beauty >= bd.beauty_score

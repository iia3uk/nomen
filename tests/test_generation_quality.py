"""Basic generation quality gates (post novelty-search redesign)."""

from __future__ import annotations

from nomen.filters.offline import OfflineFilterBank
from nomen.generation.engine import GenerationEngine
from nomen.linguistics import brand_phonotactic_score, looks_premium_brand
from nomen.models import Candidate
from nomen.scoring.scorer import score_candidate


def test_phonotactic_rejects_garbage() -> None:
    assert brand_phonotactic_score("soafyrxi") < 88
    assert brand_phonotactic_score("xqzzt") < 50
    assert not looks_premium_brand("soafyrxi")


def test_generators_produce_names() -> None:
    eng = GenerationEngine("unit-test-seed", min_len=5, max_len=9)
    names = eng.gen_char_lm(20) + eng.gen_transformer(20) + eng.gen_experimental(20)
    assert names


def test_offline_bans_buzzwords() -> None:
    bank = OfflineFilterBank()
    assert bank.check("techforge")
    assert bank.check("opencms")
    assert bank.check("smartlogic")


def test_no_verbatim_corpus_emission() -> None:
    eng = GenerationEngine("verbatim-test")
    prov = eng.generate_all(800)
    leaked = set(prov) & set(eng.lm.brands)
    assert not leaked


def test_scoring_sets_brand_score() -> None:
    eng = GenerationEngine("score-test")
    c = Candidate(name="veloran", generator="test")
    score_candidate(c, eng.lm)
    assert 0 <= c.scores.brand_score <= 100
    assert c.scores.phonetic_root
    assert c.scores.cv_pattern

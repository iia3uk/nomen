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


def test_cull_caps_cross_generator_endings() -> None:
    eng = GenerationEngine("cull-endings")
    prov = {f"na{i:04d}on": "transformer" for i in range(80)}
    prov.update({f"xy{i:04d}ka": "phoneme" for i in range(20)})
    out = eng._cull_saturated_endings(prov, max_ending_share=0.08)
    on_count = sum(1 for name in out if name.endswith("on"))
    cap = max(8, int(len(prov) * 0.08))
    assert on_count <= cap
    assert any(name.endswith("ka") for name in out)


def test_ending_pressure_blocks_flood() -> None:
    eng = GenerationEngine("ending-cap")
    eng._ending_cap = 99
    probe = next(
        (f"ko{ch}en" for ch in "bcdfghkmptvwz" if eng._accept(f"ko{ch}en")),
        None,
    )
    assert probe is not None
    eng._ending_cap = 3
    eng._ending_pressure[probe[-2:]] = 3
    assert not eng._accept(probe)

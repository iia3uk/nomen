"""Diversity / novelty-search hard constraints."""

from __future__ import annotations

from nomen.diversity.clustering import cluster_candidates, same_family
from nomen.diversity.features import phonetic_root
from nomen.diversity.novelty import NoveltyArchive
from nomen.diversity.selector import DiversitySelector
from nomen.generation.engine import GenerationEngine
from nomen.models import Candidate
from nomen.scoring.scorer import score_candidate


def _cand(name: str, gen: str = "experimental") -> Candidate:
    c = Candidate(name=name, generator=gen)
    score_candidate(c)
    return c


def test_same_family_detects_pler_cluster() -> None:
    assert same_family("plerasta", "plerora")
    assert same_family("plerasta", "plerenda")
    assert same_family("seralea", "seralio")
    assert not same_family("plerasta", "voryx")
    assert not same_family("nestra", "quivon")
    # 5-letter distance-2 pairs must not collapse (old lev<=2 over-merged)
    assert not same_family("vireo", "viran")
    assert not same_family("nesta", "nesko")


def test_cluster_rep_is_beauty_led() -> None:
    pretty = _cand("plerasta")
    pretty.scores.beauty_score = 92
    pretty.scores.overall = 88
    pretty.scores.novelty_score = 40
    pretty.scores.brand_score = 80
    novel = _cand("plerora")
    novel.scores.beauty_score = 60
    novel.scores.overall = 70
    novel.scores.novelty_score = 99
    novel.scores.brand_score = 80
    clusters = cluster_candidates([pretty, novel])
    assert len(clusters) == 1
    assert clusters[0].representative.name == "plerasta"


def test_selector_keeps_one_per_family() -> None:
    names = [
        "plerasta",
        "plerora",
        "plerenda",
        "plerenta",
        "voryx",
        "nestra",
        "quivon",
        "draxel",
        "lumior",
        "kavix",
        "zendao",
        "torvik",
    ]
    gens = ["transformer", "phoneme", "entropy", "char_lm", "experimental", "transition_graph"]
    cands = [_cand(n, gens[i % len(gens)]) for i, n in enumerate(names)]
    sel = DiversitySelector(NoveltyArchive())
    result = sel.select(cands, limit=10, winners=[])
    selected_names = [c.name for c in result.selected]
    # At most one pler*
    pler = [n for n in selected_names if n.startswith("pler")]
    assert len(pler) <= 1
    # Independent roots survive
    assert any(n in selected_names for n in ("voryx", "nestra", "quivon", "draxel", "lumior", "kavix"))


def test_novelty_penalizes_near_winners() -> None:
    arch = NoveltyArchive()
    near = arch.novelty_score("plerasta", winners=["plerora", "plerenda"])
    far = arch.novelty_score("voryx", winners=["plerora", "plerenda"])
    assert far > near


def test_generator_batch_has_multiple_sources() -> None:
    eng = GenerationEngine("diversity-seed-42")
    prov = eng.generate_all(2000)
    sources = set(prov.values())
    assert len(sources) >= 5
    # No single generator > 35% of raw batch (quota soft at generation)
    from collections import Counter

    counts = Counter(prov.values())
    for g, n in counts.items():
        assert n / len(prov) <= 0.40, f"{g} dominated batch with {n}/{len(prov)}"


def test_evolution_injection_mix_runs() -> None:
    eng = GenerationEngine("evo-mix")
    eng.set_archive(["figma", "vercel", "notion", "stripe", "raycast", "linear"])
    out = eng.gen_evolutionary(80)
    assert len(out) >= 10


def test_reseed_changes_region() -> None:
    eng = GenerationEngine("reseed-test")
    a = eng.generate_all(200)
    eng.reseed_exploration("test")
    b = eng.generate_all(200)
    # Should not be identical sets after reseed
    assert set(a) != set(b)


def test_phonetic_roots_differ_for_independent_brands() -> None:
    roots = {phonetic_root(n) for n in ("voryx", "nestra", "quivon", "draxel", "lumior")}
    assert len(roots) >= 4

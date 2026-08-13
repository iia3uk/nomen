"""BrandScore + BeautyScore + CollisionScore. Novelty filled by DiversitySelector."""

from __future__ import annotations

from nomen.diversity.features import cv_pattern, phonetic_root
from nomen.linguistics import brand_phonotactic_score, is_vowel, max_consonant_run
from nomen.models import Candidate, Scores
from nomen.scoring.beauty import beauty_breakdown
from nomen.scoring.collision import collision_score
from nomen.scoring.overall import compute_overall
from nomen.training.models import BrandLanguageModels

_HOME = set("asdfghjklqwertyuiop")
_LEFT = set("qwertasdfgzxcvb")


def score_candidate(candidate: Candidate, lm: BrandLanguageModels | None = None) -> Candidate:
    n = candidate.name
    pronounce = brand_phonotactic_score(n)

    mem = 70.0
    if 5 <= len(n) <= 7:
        mem += 18
    elif len(n) == 8:
        mem += 10
    vowels = sum(1 for c in n if is_vowel(c))
    mem += min(12, vowels * 3)
    if n.endswith(("a", "o", "y", "e", "n", "x", "v")):
        mem += 5

    easy = sum(1 for c in n if c in _HOME) / max(len(n), 1)
    alt = sum(1 for i in range(len(n) - 1) if (n[i] in _LEFT) != (n[i + 1] in _LEFT))
    typing = 55 + easy * 30 + min(12, alt * 2)

    asc = set("bdfhiklt")
    desc = set("gjpqy")
    a = sum(1 for c in n if c in asc)
    d = sum(1 for c in n if c in desc)
    visual = 68 + (len(n) - a - d) * 3
    if a and d:
        visual += 8
    if max_consonant_run(n) <= 2:
        visual += 6

    brand = 72.0
    premium = 70.0
    if n[0] in "vkmfpzqjw":
        brand += 8
        premium += 6
    if n.endswith(("el", "en", "on", "an", "ay", "ix", "ex", "o", "a", "y", "x")):
        brand += 10
        premium += 10
    if n.endswith(("er", "ar", "or")) and n.count("r") + n.count("l") >= 2:
        brand -= 12
        premium -= 14
    if lm is not None:
        lp = lm.char_lm_logprob(n)
        brand += max(-8, min(12, (lp + len(n) * 2.8) * 3))
        premium += max(-8, min(10, (lp + len(n) * 2.8) * 2.5))

    intl = 88.0
    for bad in ("th", "wh", "gh", "ck"):
        if bad in n:
            intl -= 7
    if max_consonant_run(n) >= 3:
        intl -= 15

    collision_prob = 18.0 + max(0, 8 - len(n)) * 7
    seo = 100.0 - collision_prob * 0.6

    brand_score = (
        pronounce * 0.20
        + mem * 0.14
        + typing * 0.08
        + visual * 0.10
        + brand * 0.18
        + premium * 0.18
        + intl * 0.12
    )

    # Independent aesthetic model — no registries / SEO / collisions
    bd = beauty_breakdown(n)
    coll = collision_score(n)

    # Temporary overall without novelty (novelty filled later)
    overall = compute_overall(
        beauty=bd.beauty_score,
        brand=brand_score,
        novelty=50.0,
        collision=coll,
    )

    candidate.scores = Scores(
        pronounceability=round(pronounce, 2),
        memorability=round(min(100, mem), 2),
        typing_speed=round(min(100, typing), 2),
        visual_balance=round(min(100, visual), 2),
        brand_strength=round(min(100, max(0, brand)), 2),
        premium_feel=round(min(100, max(0, premium)), 2),
        international_readability=round(min(100, max(0, intl)), 2),
        collision_probability=round(min(100, max(0, collision_prob)), 2),
        seo_uniqueness=round(min(100, max(0, seo)), 2),
        beauty_score=bd.beauty_score,
        brand_score=round(min(100, max(0, brand_score)), 2),
        novelty_score=0.0,
        collision_score=coll,
        overall=overall,
        cv_pattern=cv_pattern(n),
        phonetic_root=phonetic_root(n),
    )
    candidate.meta["beauty"] = {
        "readability": bd.readability,
        "memorability": bd.memorability,
        "premium": bd.premium,
        "naturalness": bd.naturalness,
        "rejects": list(bd.reject_reasons),
    }
    return candidate

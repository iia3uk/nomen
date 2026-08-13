"""Multi-metric clustering — one representative per phonetic/visual family."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import JaroWinkler, Levenshtein

from nomen.diversity.features import (
    bigram_vector,
    cosine,
    metaphone_pair,
    phonetic_root,
    visual_similarity,
)
from nomen.linguistics import normalize
from nomen.models import Candidate


@dataclass
class Cluster:
    representative: Candidate
    members: list[Candidate] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.members:
            self.members = [self.representative]


def _rank(c: Candidate) -> tuple[float, float, float, float]:
    """Beauty-led: overall/beauty first — novelty must not pick the family rep."""
    return (
        c.scores.overall,
        c.scores.beauty_score,
        c.scores.brand_score,
        c.scores.novelty_score,
    )


def _lev_cap(x: str, y: str) -> int:
    m = min(len(x), len(y))
    if m <= 6:
        return 1
    return 2


def same_family(a: str, b: str) -> bool:
    """True if names belong to one phonetic / visual root family."""
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y:
        return True
    short = min(len(x), len(y)) <= 6
    jw = JaroWinkler.similarity(x, y)
    lev = Levenshtein.distance(x, y)

    if phonetic_root(x) and phonetic_root(x) == phonetic_root(y):
        # Same metaphone/root — still require some string nearness to avoid over-merge
        jw_need = 0.82 if short else 0.72
        if jw >= jw_need or lev <= (2 if short else 3):
            return True
    if metaphone_pair(x)[0] and metaphone_pair(x)[0] == metaphone_pair(y)[0]:
        if abs(len(x) - len(y)) <= 2 and jw >= (0.86 if short else 0.78):
            return True
    if lev <= _lev_cap(x, y):
        return True
    if jw >= (0.94 if short else 0.90):
        return True
    if visual_similarity(x, y) >= (0.92 if short else 0.86):
        return True
    if cosine(bigram_vector(x), bigram_vector(y)) >= 0.82 and abs(len(x) - len(y)) <= 2:
        if not short:
            return True
    # Shared long prefix (plerasta/plerora) — 4 letters is too much of a 5–6 letter name
    prefix = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            prefix += 1
        else:
            break
    if prefix >= 5:
        return True
    if prefix >= 4 and min(len(x), len(y)) >= 7 and abs(len(x) - len(y)) <= 3:
        return True
    return False


def _place(cand: Candidate, clusters: list[Cluster]) -> None:
    for cl in clusters:
        if same_family(cand.name, cl.representative.name):
            cl.members.append(cand)
            if _rank(cand) > _rank(cl.representative):
                cl.representative = cand
            return
    clusters.append(Cluster(representative=cand))


def cluster_candidates(candidates: list[Candidate]) -> list[Cluster]:
    """Greedy clustering by phonetic-root buckets, then a cheap cross-root merge."""
    ordered = sorted(candidates, key=_rank, reverse=True)
    by_root: dict[str, list[Candidate]] = defaultdict(list)
    for cand in ordered:
        by_root[phonetic_root(cand.name) or "_"].append(cand)

    buckets: list[Cluster] = []
    for group in by_root.values():
        local: list[Cluster] = []
        for cand in group:
            _place(cand, local)
        buckets.extend(local)

    merged: list[Cluster] = []
    for cl in buckets:
        placed = False
        for m in merged:
            if same_family(cl.representative.name, m.representative.name):
                m.members.extend(x for x in cl.members if x.name not in {c.name for c in m.members})
                if _rank(cl.representative) > _rank(m.representative):
                    m.representative = cl.representative
                placed = True
                break
        if not placed:
            merged.append(cl)
    return merged

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


def _prefix_len(x: str, y: str) -> int:
    n = 0
    for a, b in zip(x, y):
        if a != b:
            break
        n += 1
    return n


def same_family(a: str, b: str) -> bool:
    """True if names belong to one phonetic / visual root family."""
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y:
        return True

    prefix = _prefix_len(x, y)
    if prefix >= 5:
        return True
    if prefix >= 4 and min(len(x), len(y)) >= 7 and abs(len(x) - len(y)) <= 3:
        return True
    if abs(len(x) - len(y)) > 3:
        return False

    short = min(len(x), len(y)) <= 6
    jw = JaroWinkler.similarity(x, y)
    lev = Levenshtein.distance(x, y)

    x_root, y_root = phonetic_root(x), phonetic_root(y)
    if x_root and x_root == y_root:
        # Same metaphone/root — still require some string nearness to avoid over-merge
        jw_need = 0.82 if short else 0.72
        if jw >= jw_need or lev <= (2 if short else 3):
            return True
    x_mp, y_mp = metaphone_pair(x)[0], metaphone_pair(y)[0]
    if x_mp and x_mp == y_mp:
        if abs(len(x) - len(y)) <= 2 and jw >= (0.86 if short else 0.78):
            return True
    if lev <= _lev_cap(x, y):
        return True
    if jw >= (0.94 if short else 0.90):
        return True
    lev_sim = 1.0 - lev / max(len(x), len(y), 1)
    prefix_ratio = prefix / max(min(len(x), len(y)), 1)
    visual = 0.45 * jw + 0.35 * lev_sim + 0.20 * prefix_ratio
    if visual >= (0.92 if short else 0.86):
        return True
    if cosine(bigram_vector(x), bigram_vector(y)) >= 0.82 and abs(len(x) - len(y)) <= 2:
        if not short:
            return True
    return False


def _blocking_keys(name: str) -> tuple[str, ...]:
    """High-recall blocks for cross-root merge — false positives are filtered by same_family."""
    n = normalize(name)
    keys = [f"p3:{n[:3]}"]
    if len(n) >= 4:
        keys.append(f"p3:{n[1:4]}")
    mp = metaphone_pair(n)[0]
    if mp:
        keys.append(f"mp:{mp[:4].lower()}")
    if len(n) >= 2:
        keys.append(f"L{len(n)}:{n[:2]}")
    return tuple(keys)


def _place(cand: Candidate, clusters: list[Cluster]) -> None:
    for cl in clusters:
        if same_family(cand.name, cl.representative.name):
            cl.members.append(cand)
            if _rank(cand) > _rank(cl.representative):
                cl.representative = cand
            return
    clusters.append(Cluster(representative=cand))


def cluster_candidates(candidates: list[Candidate]) -> list[Cluster]:
    """Greedy clustering by phonetic-root buckets, then a blocked cross-root merge."""
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
    index: dict[str, list[Cluster]] = defaultdict(list)
    for cl in buckets:
        probes: list[Cluster] = []
        seen: set[int] = set()
        for key in _blocking_keys(cl.representative.name):
            for m in index.get(key, []):
                mid = id(m)
                if mid not in seen:
                    seen.add(mid)
                    probes.append(m)
        placed = False
        for m in probes:
            if same_family(cl.representative.name, m.representative.name):
                have = {c.name for c in m.members}
                m.members.extend(x for x in cl.members if x.name not in have)
                if _rank(cl.representative) > _rank(m.representative):
                    m.representative = cl.representative
                placed = True
                break
        if not placed:
            merged.append(cl)
            for key in _blocking_keys(cl.representative.name):
                index[key].append(cl)
    return merged

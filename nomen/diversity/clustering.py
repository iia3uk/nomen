"""Multi-metric clustering — one representative per phonetic/visual family."""

from __future__ import annotations

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


def same_family(a: str, b: str) -> bool:
    """True if names belong to one phonetic / visual root family."""
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return False
    if x == y:
        return True
    if phonetic_root(x) and phonetic_root(x) == phonetic_root(y):
        # Same metaphone/root — still require some string nearness to avoid over-merge
        if JaroWinkler.similarity(x, y) >= 0.72 or Levenshtein.distance(x, y) <= 3:
            return True
    if metaphone_pair(x)[0] and metaphone_pair(x)[0] == metaphone_pair(y)[0]:
        if abs(len(x) - len(y)) <= 2 and JaroWinkler.similarity(x, y) >= 0.78:
            return True
    if Levenshtein.distance(x, y) <= 2:
        return True
    if JaroWinkler.similarity(x, y) >= 0.90:
        return True
    if visual_similarity(x, y) >= 0.86:
        return True
    if cosine(bigram_vector(x), bigram_vector(y)) >= 0.82 and abs(len(x) - len(y)) <= 2:
        return True
    # Shared long prefix (plerasta/plerora)
    prefix = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            prefix += 1
        else:
            break
    if prefix >= 4 and abs(len(x) - len(y)) <= 3:
        return True
    return False


def cluster_candidates(candidates: list[Candidate]) -> list[Cluster]:
    """Greedy clustering; keep highest overall as representative."""
    ordered = sorted(
        candidates,
        key=lambda c: (c.scores.novelty_score, c.scores.brand_score, c.scores.overall),
        reverse=True,
    )
    clusters: list[Cluster] = []
    for cand in ordered:
        placed = False
        for cl in clusters:
            if same_family(cand.name, cl.representative.name):
                cl.members.append(cand)
                # Prefer higher combined score as representative
                if (cand.scores.novelty_score, cand.scores.brand_score) > (
                    cl.representative.scores.novelty_score,
                    cl.representative.scores.brand_score,
                ):
                    cl.representative = cand
                placed = True
                break
        if not placed:
            clusters.append(Cluster(representative=cand))
    return clusters

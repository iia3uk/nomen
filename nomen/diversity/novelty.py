"""Novelty score — distance from archive / population / brands (vectorized + parallel)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

from nomen.diversity.features import (
    bigram_vector,
    brand_archive,
    char_embedding,
    cosine,
    metaphone_pair,
    phonetic_root,
    visual_similarity,
)
from nomen.linguistics import normalize
from nomen.parallel import parallel_map


@dataclass(slots=True)
class _Ref:
    name: str
    root: str
    meta: str
    bigram: np.ndarray
    chars: np.ndarray


class NoveltyArchive:
    """Behavioral archive for novelty search."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._name_set: set[str] = set()
        self.roots: list[str] = []
        self.bigrams: list[np.ndarray] = []
        self.chars: list[np.ndarray] = []
        self.metas: list[str] = []
        for b in sorted(brand_archive()):
            if 4 <= len(b) <= 12:
                self.add(b)

    def add(self, name: str) -> None:
        n = normalize(name)
        if not n or n in self._name_set:
            return
        self._name_set.add(n)
        self.names.append(n)
        self.roots.append(phonetic_root(n))
        self.bigrams.append(bigram_vector(n))
        self.chars.append(char_embedding(n))
        self.metas.append(metaphone_pair(n)[0])

    def extend(self, names: Sequence[str]) -> None:
        for n in names:
            self.add(n)

    def _refs(
        self,
        *,
        population: Sequence[str] | None,
        winners: Sequence[str] | None,
        max_refs: int = 400,
    ) -> list[_Ref]:
        """Build a capped, feature-precomputed reference set."""
        # Prefer recent archive + winners + sampled population
        ordered: list[str] = []
        ordered.extend(self.names[-1500:])
        if winners:
            ordered.extend(normalize(x) for x in winners if x)
        if population:
            # Stratified sample of population — don't explode to O(n²)
            pop = [normalize(x) for x in population if x]
            if len(pop) > 300:
                step = max(1, len(pop) // 300)
                pop = pop[::step]
            ordered.extend(pop)

        uniq = list(dict.fromkeys(n for n in ordered if n))
        if len(uniq) > max_refs:
            step = max(1, len(uniq) // max_refs)
            uniq = uniq[::step][:max_refs]

        # Reuse archive features when possible
        idx = {n: i for i, n in enumerate(self.names)}
        refs: list[_Ref] = []
        for n in uniq:
            i = idx.get(n)
            if i is not None:
                refs.append(
                    _Ref(
                        name=n,
                        root=self.roots[i],
                        meta=self.metas[i],
                        bigram=self.bigrams[i],
                        chars=self.chars[i],
                    )
                )
            else:
                refs.append(
                    _Ref(
                        name=n,
                        root=phonetic_root(n),
                        meta=metaphone_pair(n)[0],
                        bigram=bigram_vector(n),
                        chars=char_embedding(n),
                    )
                )
        return refs

    def novelty_score(
        self,
        name: str,
        *,
        population: Sequence[str] | None = None,
        winners: Sequence[str] | None = None,
        k: int = 8,
        _refs: list[_Ref] | None = None,
    ) -> float:
        n = normalize(name)
        refs = _refs if _refs is not None else self._refs(population=population, winners=winners)
        refs = [r for r in refs if r.name != n]
        if not refs:
            return 100.0

        nb = bigram_vector(n)
        nc = char_embedding(n)
        nm = metaphone_pair(n)[0]
        nroot = phonetic_root(n)

        distances: list[float] = []
        for other in refs:
            lev = Levenshtein.distance(n, other.name) / max(len(n), len(other.name), 1)
            jw = 1.0 - JaroWinkler.similarity(n, other.name)
            # Cheap visual proxy without full visual_similarity when far
            if lev > 0.55 and jw > 0.45:
                vis = 0.7
            else:
                vis = 1.0 - visual_similarity(n, other.name)
            bg = 1.0 - cosine(nb, other.bigram)
            ch = 1.0 - cosine(nc, other.chars)
            meta = 0.0 if (nm and nm == other.meta) else 1.0
            root = 0.0 if nroot and nroot == other.root else 1.0
            d = (
                0.22 * lev
                + 0.18 * jw
                + 0.18 * vis
                + 0.14 * bg
                + 0.10 * ch
                + 0.10 * meta
                + 0.08 * root
            )
            distances.append(d)

        distances.sort()
        avg = sum(distances[:k]) / max(len(distances[:k]), 1)
        score = (avg - 0.12) / 0.55 * 100.0
        if winners:
            for w in winners:
                wn = normalize(w)
                if phonetic_root(wn) == nroot and nroot:
                    score -= 55.0
                if visual_similarity(n, wn) >= 0.82:
                    score -= 40.0
        return float(max(0.0, min(100.0, score)))

    def novelty_scores_batch(
        self,
        names: Sequence[str],
        *,
        population: Sequence[str] | None = None,
        winners: Sequence[str] | None = None,
        k: int = 8,
        workers: int | None = None,
    ) -> list[float]:
        """Parallel novelty scoring with shared precomputed references."""
        refs = self._refs(population=population, winners=winners)
        win = list(winners or [])

        def _one(name: str) -> float:
            return self.novelty_score(name, winners=win, k=k, _refs=refs)

        return parallel_map(_one, list(names), workers=workers, chunksize=64)

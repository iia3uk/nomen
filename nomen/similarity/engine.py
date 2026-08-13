"""Multi-metric near-match rejection against known brands."""

from __future__ import annotations

from typing import Any

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

from nomen.config import SimilarityConfig
from nomen.linguistics import occupied_brands, normalize
from nomen.models import Candidate, Stage

try:
    from metaphone import doublemetaphone
except Exception:  # pragma: no cover

    def doublemetaphone(s: str) -> tuple[str, str]:  # type: ignore[misc]
        return s[:4].upper(), ""


def soundex(word: str) -> str:
    word = normalize(word)
    if not word:
        return ""
    first = word[0].upper()
    mapping = {
        **dict.fromkeys(list("bfpv"), "1"),
        **dict.fromkeys(list("cgjkqsxz"), "2"),
        **dict.fromkeys(list("dt"), "3"),
        **dict.fromkeys(list("l"), "4"),
        **dict.fromkeys(list("mn"), "5"),
        **dict.fromkeys(list("r"), "6"),
    }
    digits: list[str] = []
    prev = mapping.get(word[0], "")
    for ch in word[1:]:
        d = mapping.get(ch, "")
        if d and d != prev:
            digits.append(d)
        prev = d if d else prev if ch in "hw" else ""
    return (first + "".join(digits) + "000")[:4]


class SimilarityEngine:
    def __init__(self, cfg: SimilarityConfig, embedding_model: str | None = None) -> None:
        self.cfg = cfg
        # Full collision corpus (generation trains on premium only, never emits these)
        self.corpus = sorted(occupied_brands())
        self.by_len: dict[int, list[str]] = {}
        for w in self.corpus:
            self.by_len.setdefault(len(w), []).append(w)
        self._embedder = None
        self._corpus_emb: np.ndarray | None = None
        self._embedding_model = embedding_model
        self._metaphone_index: dict[str, list[str]] = {}
        for w in self.corpus:
            p1, p2 = doublemetaphone(w)
            for p in (p1, p2):
                if p:
                    self._metaphone_index.setdefault(p, []).append(w)

    def enable_embeddings(self, enabled: bool) -> None:
        if not enabled or not self._embedding_model:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self._embedding_model)
            self._corpus_emb = self._embedder.encode(
                self.corpus, normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            self._embedder = None
            self._corpus_emb = None

    def _ngram_sim(self, a: str, b: str, n: int = 3) -> float:
        def grams(s: str) -> set[str]:
            if len(s) < n:
                return {s}
            return {s[i : i + n] for i in range(len(s) - n + 1)}

        ga, gb = grams(a), grams(b)
        if not ga or not gb:
            return 0.0
        return len(ga & gb) / len(ga | gb)

    def _char_lm_distance(self, a: str, b: str) -> float:
        # Symmetric bigram Jaccard distance
        def bigrams(s: str) -> set[str]:
            return {s[i : i + 2] for i in range(len(s) - 1)} or {s}

        ga, gb = bigrams(a), bigrams(b)
        inter = len(ga & gb)
        union = len(ga | gb) or 1
        return 1.0 - inter / union

    def reason(self, name: str) -> str | None:
        n = normalize(name)
        # Stem / prefix collisions against known brands (replicate→replican)
        for other in self.corpus:
            if len(other) >= 5 and len(n) >= 5:
                if n.startswith(other[:5]) or other.startswith(n[:5]):
                    if abs(len(n) - len(other)) <= 3 or n[:6] == other[:6]:
                        return f"stem/prefix collision with '{other}'"
                if other in n or n in other:
                    return f"substring collision with '{other}'"

        lengths = range(max(3, len(n) - 2), len(n) + 3)
        candidates: list[str] = []
        for L in lengths:
            candidates.extend(self.by_len.get(L, []))

        if self.cfg.metaphone_match:
            p1, p2 = doublemetaphone(n)
            for p in (p1, p2):
                if p:
                    candidates.extend(self._metaphone_index.get(p, []))

        seen: set[str] = set()
        for other in candidates:
            if other in seen or other == n:
                if other == n:
                    return f"exact corpus match '{other}'"
                continue
            seen.add(other)

            lev = Levenshtein.distance(n, other)
            if lev <= self.cfg.levenshtein_max:
                return f"Levenshtein {lev} vs '{other}'"

            dam = DamerauLevenshtein.distance(n, other)
            if dam <= self.cfg.damerau_max:
                return f"Damerau-Levenshtein {dam} vs '{other}'"

            jw = JaroWinkler.similarity(n, other)
            if jw >= self.cfg.jaro_winkler_min:
                return f"Jaro-Winkler {jw:.3f} vs '{other}'"

            ng = self._ngram_sim(n, other)
            if ng >= self.cfg.ngram_min:
                return f"n-gram {ng:.3f} vs '{other}'"

            if soundex(n) == soundex(other) and abs(len(n) - len(other)) <= 2 and jw > 0.75:
                return f"Soundex match vs '{other}'"

            if self.cfg.metaphone_match:
                a1, a2 = doublemetaphone(n)
                b1, b2 = doublemetaphone(other)
                if a1 and a1 == b1 and abs(len(n) - len(other)) <= 2:
                    return f"Double Metaphone '{a1}' vs '{other}'"

            if self._char_lm_distance(n, other) < 0.35 and jw > 0.8:
                return f"char-language near '{other}'"

            # Token set ratio as soft embedding-free cosine analogue
            if fuzz.ratio(n, other) >= 90:
                return f"ratio {fuzz.ratio(n, other)} vs '{other}'"

        if self._embedder is not None and self._corpus_emb is not None:
            vec = self._embedder.encode([n], normalize_embeddings=True)[0]
            sims = self._corpus_emb @ vec
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            if sim >= self.cfg.embedding_cosine_min:
                return f"embedding cosine {sim:.3f} vs '{self.corpus[idx]}'"

        return None

    def apply(self, candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        ok: list[Candidate] = []
        bad: list[Candidate] = []
        for cand in candidates:
            r = self.reason(cand.name)
            if r:
                bad.append(cand.reject(Stage.SIMILARITY, r))
            else:
                ok.append(cand)
        return ok, bad

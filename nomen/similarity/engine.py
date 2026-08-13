"""Multi-metric near-match rejection against known brands."""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein, JaroWinkler, Levenshtein

from nomen.config import SimilarityConfig
from nomen.linguistics import occupied_brands, normalize
from nomen.models import Candidate, Stage
from nomen.parallel import parallel_map

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

    def _structural_collision(self, n: str) -> str | None:
        if len(n) < 5:
            return None
        n5, n6 = n[:5], n[:6]
        lo = max(5, len(n) - 3)
        hi = len(n) + 3
        for L in range(lo, hi + 1):
            for other in self.by_len.get(L, []):
                if n.startswith(other[:5]) or other.startswith(n5):
                    if abs(len(n) - L) <= 3 or (len(other) >= 6 and n6 == other[:6]):
                        return f"stem/prefix collision with '{other}'"
                if other in n or n in other:
                    return f"substring collision with '{other}'"
        for L, bucket in self.by_len.items():
            if L < 5 or lo <= L <= hi:
                continue
            if L < len(n):
                for other in bucket:
                    if other in n:
                        return f"substring collision with '{other}'"
            else:
                for other in bucket:
                    if n in other:
                        return f"substring collision with '{other}'"
        return None

    def _pair_reason(
        self,
        n: str,
        other: str,
        *,
        n_sx: str,
        n_mp: tuple[str, str],
        check_metaphone: bool,
    ) -> str | None:
        if other == n:
            return f"exact corpus match '{other}'"

        far_cut = max(4, self.cfg.levenshtein_max + 2)
        lev = Levenshtein.distance(n, other, score_cutoff=far_cut + 1)
        if lev <= self.cfg.levenshtein_max:
            return f"Levenshtein {lev} vs '{other}'"

        if lev <= self.cfg.damerau_max + 1:
            dam = DamerauLevenshtein.distance(n, other)
            if dam <= self.cfg.damerau_max:
                return f"Damerau-Levenshtein {dam} vs '{other}'"

        close = lev <= far_cut
        len_close = abs(len(n) - len(other)) <= 2
        jw = JaroWinkler.similarity(n, other) if close or len_close else 0.0
        if jw >= self.cfg.jaro_winkler_min:
            return f"Jaro-Winkler {jw:.3f} vs '{other}'"

        if close:
            ng = self._ngram_sim(n, other)
            if ng >= self.cfg.ngram_min:
                return f"n-gram {ng:.3f} vs '{other}'"
            if self._char_lm_distance(n, other) < 0.35 and jw > 0.8:
                return f"char-language near '{other}'"
            ratio = fuzz.ratio(n, other)
            if ratio >= 90:
                return f"ratio {ratio} vs '{other}'"

        if len_close:
            if n_sx == soundex(other) and jw > 0.75:
                return f"Soundex match vs '{other}'"
            if check_metaphone and self.cfg.metaphone_match:
                a1 = n_mp[0]
                b1, _b2 = doublemetaphone(other)
                if a1 and a1 == b1:
                    return f"Double Metaphone '{a1}' vs '{other}'"
        return None

    def reason(self, name: str) -> str | None:
        n = normalize(name)
        hit = self._structural_collision(n)
        if hit:
            return hit

        n_sx = soundex(n)
        n_mp = doublemetaphone(n) if self.cfg.metaphone_match else ("", "")
        seen: set[str] = set()

        def consider(others: list[str], *, check_metaphone: bool) -> str | None:
            for other in others:
                if other in seen:
                    continue
                seen.add(other)
                r = self._pair_reason(
                    n, other, n_sx=n_sx, n_mp=n_mp, check_metaphone=check_metaphone
                )
                if r:
                    return r
            return None

        phonetic: list[str] = []
        if self.cfg.metaphone_match:
            for p in n_mp:
                if p:
                    phonetic.extend(self._metaphone_index.get(p, []))
        hit = consider(phonetic, check_metaphone=True)
        if hit:
            return hit

        window: list[str] = []
        for L in range(max(3, len(n) - 2), len(n) + 3):
            window.extend(self.by_len.get(L, []))
        hit = consider(window, check_metaphone=False)
        if hit:
            return hit

        if self._embedder is not None and self._corpus_emb is not None:
            vec = self._embedder.encode([n], normalize_embeddings=True)[0]
            sims = self._corpus_emb @ vec
            idx = int(np.argmax(sims))
            sim = float(sims[idx])
            if sim >= self.cfg.embedding_cosine_min:
                return f"embedding cosine {sim:.3f} vs '{self.corpus[idx]}'"

        return None

    def apply(self, candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
        reasons = parallel_map(self.reason, [c.name for c in candidates])
        ok: list[Candidate] = []
        bad: list[Candidate] = []
        for cand, r in zip(candidates, reasons, strict=True):
            if r:
                bad.append(cand.reject(Stage.SIMILARITY, r))
            else:
                ok.append(cand)
        return ok, bad

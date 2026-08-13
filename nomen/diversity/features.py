"""Phonetic roots, CV patterns, embeddings for diversity metrics."""

from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

from nomen.linguistics import is_vowel, normalize, occupied_brands

try:
    from metaphone import doublemetaphone
except Exception:  # pragma: no cover

    def doublemetaphone(s: str) -> tuple[str, str]:  # type: ignore[misc]
        return s[:4].upper(), ""


_VOWELS = "aeiouy"


def cv_pattern(name: str) -> str:
    n = normalize(name)
    return "".join("V" if is_vowel(c) else "C" for c in n)


def phonetic_root(name: str) -> str:
    """
    Stable family key for a coined brand.

    Uses Double Metaphone primary code when available, else onset+nucleus stub.
    """
    n = normalize(name)
    if not n:
        return ""
    primary, _ = doublemetaphone(n)
    if primary and len(primary) >= 2:
        return primary[:4].lower()
    # Fallback: first consonant run + first vowel + next consonant
    m = re.match(r"^([^aeiouy]{0,2}[aeiouy][^aeiouy]?).+", n)
    if m:
        return m.group(1)
    return n[:3]


def metaphone_pair(name: str) -> tuple[str, str]:
    a, b = doublemetaphone(normalize(name))
    return a or "", b or ""


def bigram_vector(name: str) -> np.ndarray:
    """26*26 sparse bigram bag normalized to unit length."""
    n = normalize(name)
    vec = np.zeros(26 * 26, dtype=np.float64)
    if len(n) < 2:
        return vec
    for i in range(len(n) - 1):
        a, b = ord(n[i]) - 97, ord(n[i + 1]) - 97
        if 0 <= a < 26 and 0 <= b < 26:
            vec[a * 26 + b] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def char_embedding(name: str) -> np.ndarray:
    """Lightweight positional character embedding (length 26 + 10 length bins)."""
    n = normalize(name)
    vec = np.zeros(36, dtype=np.float64)
    for i, c in enumerate(n):
        idx = ord(c) - 97
        if 0 <= idx < 26:
            vec[idx] += 1.0 + 0.15 * i
    if n:
        vec[26 + min(len(n), 9)] = 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def visual_similarity(a: str, b: str) -> float:
    """High when names look like spelling variants (same length-ish + shared prefix/suffix)."""
    x, y = normalize(a), normalize(b)
    if not x or not y:
        return 0.0
    jw = JaroWinkler.similarity(x, y)
    lev = Levenshtein.distance(x, y)
    lev_sim = 1.0 - lev / max(len(x), len(y), 1)
    prefix = 0.0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            prefix += 1
        else:
            break
    prefix_ratio = prefix / max(min(len(x), len(y)), 1)
    return 0.45 * jw + 0.35 * lev_sim + 0.20 * prefix_ratio


def letter_histogram(name: str) -> dict[str, float]:
    n = normalize(name)
    if not n:
        return {}
    total = len(n)
    return {c: n.count(c) / total for c in set(n)}


@lru_cache(maxsize=1)
def brand_archive() -> frozenset[str]:
    return occupied_brands()


def soft_consonant_ratio(name: str) -> float:
    n = normalize(name)
    if not n:
        return 0.0
    return sum(n.count(c) for c in "rlns") / len(n)


def vowel_a_ratio(name: str) -> float:
    n = normalize(name)
    if not n:
        return 0.0
    return n.count("a") / len(n)


def entropy_bits(name: str) -> float:
    n = normalize(name)
    if not n:
        return 0.0
    counts: dict[str, int] = {}
    for c in n:
        counts[c] = counts.get(c, 0) + 1
    h = 0.0
    for v in counts.values():
        p = v / len(n)
        h -= p * math.log2(p)
    return h

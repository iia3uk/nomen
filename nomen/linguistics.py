"""Phonotactics, premium brand shape, and corpus utilities."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from nomen.config import DATA_DIR

_VOWELS = frozenset("aeiouy")
_ALPHA = re.compile(r"[^a-z]")

_PREMIUM_ENDINGS = (
    "el", "en", "er", "ar", "or", "an", "on", "in",
    "is", "us", "ay", "ey", "ie", "ia", "a", "o", "y", "e",
)

_HARD_CLUSTERS = frozenset(
    {
        "bx", "cx", "dx", "fx", "gx", "hx", "jx", "kx", "mx", "nx", "px", "qx",
        "sx", "tx", "vx", "wx", "xx", "zx", "bq", "cq", "dq", "fq", "gq", "hq",
        "jq", "kq", "lq", "mq", "nq", "pq", "qq", "rq", "sq", "tq", "vq", "wq",
        "xq", "yq", "zq", "gk", "kp", "tk", "tp", "pk", "bg", "gd", "tb", "pb",
        "vf", "fv", "zs", "sz", "xr", "xl", "xw", "wz", "zw", "yy", "uu", "ii",
        "aa", "ooo", "fy", "yi", "yx", "xy", "wx", "kx", "mx", "pz", "qt",
        "ss", "rrr", "lll",
    }
)

_ALLOWED_ONSETS = frozenset(
    {
        "b", "bl", "br", "c", "ch", "cl", "cr", "d", "dr", "f", "fl", "fr",
        "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "pl",
        "pr", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "t",
        "th", "tr", "v", "w", "z",
    }
)


def normalize(text: str) -> str:
    return _ALPHA.sub("", text.lower())


@lru_cache(maxsize=16)
def load_lines(filename: str) -> frozenset[str]:
    path = DATA_DIR / filename
    if not path.exists():
        return frozenset()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        w = normalize(line)
        if w:
            out.add(w)
    return frozenset(out)


@lru_cache(maxsize=1)
def occupied_brands() -> frozenset[str]:
    """Names that must never be emitted: training corpus plus reserved/owned brands."""
    return (
        load_lines("brands_corpus.txt")
        | load_lines("brands_premium.txt")
        | load_lines("reserved.txt")
    )


@lru_cache(maxsize=1)
def reserved_brands() -> frozenset[str]:
    return load_lines("reserved.txt")


@lru_cache(maxsize=1)
def brand_bigrams() -> frozenset[str]:
    grams: set[str] = set()
    for w in load_lines("brands_premium.txt") or load_lines("brands_corpus.txt"):
        for i in range(len(w) - 1):
            grams.add(w[i : i + 2])
    return frozenset(grams)


@lru_cache(maxsize=1)
def brand_trigrams() -> frozenset[str]:
    grams: set[str] = set()
    for w in load_lines("brands_premium.txt") or load_lines("brands_corpus.txt"):
        for i in range(len(w) - 2):
            grams.add(w[i : i + 3])
    return frozenset(grams)


def is_vowel(ch: str) -> bool:
    return ch in _VOWELS


def max_consonant_run(name: str) -> int:
    best = cur = 0
    for ch in name:
        if not is_vowel(ch):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def max_vowel_run(name: str) -> int:
    best = cur = 0
    for ch in name:
        if is_vowel(ch):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def onset_of(name: str) -> str:
    i = 0
    while i < len(name) and not is_vowel(name[i]):
        i += 1
    return name[:i] if i else name[:1]


def brand_phonotactic_score(name: str) -> float:
    """0–100. Strict premium coined-brand gate."""
    if not name or not name.isalpha():
        return 0.0
    n = normalize(name)
    score = 100.0

    if not (5 <= len(n) <= 9):
        score -= 45
    if not any(is_vowel(c) for c in n):
        return 0.0

    cr = max_consonant_run(n)
    vr = max_vowel_run(n)
    if cr >= 3:
        score -= 35 if cr == 3 else 55
    if vr >= 3:
        score -= 35

    onset = onset_of(n)
    if onset and onset not in _ALLOWED_ONSETS:
        score -= 30

    # Every bigram must exist in real software brands
    bigs = brand_bigrams()
    rare_bi = 0
    for i in range(len(n) - 1):
        bg = n[i : i + 2]
        if bg in _HARD_CLUSTERS:
            score -= 22
        if bg not in bigs:
            rare_bi += 1
            score -= 18
    if rare_bi >= 2:
        score -= 20

    # Prefer trigrams attested in brands
    tris = brand_trigrams()
    if len(n) >= 3:
        attested = sum(1 for i in range(len(n) - 2) if n[i : i + 3] in tris)
        ratio = attested / (len(n) - 2)
        if ratio < 0.5:
            score -= 25
        elif ratio > 0.85:
            score += 8

    alt = sum(1 for i in range(len(n) - 1) if is_vowel(n[i]) != is_vowel(n[i + 1]))
    score += min(10, alt * 1.8)

    if any(n.endswith(e) for e in _PREMIUM_ENDINGS):
        score += 6
    else:
        score -= 18

    if n.endswith(("q", "j", "w", "x", "z", "c", "u", "i")):
        score -= 20
    if n[0] in "qxz":
        score -= 18

    # Weak / artificial patterns
    if n.count("er") >= 2 or n.count("ar") >= 2:
        score -= 25
    for i in range(len(n) - 1):
        if n[i] == n[i + 1] and n[i] not in "lnrs":
            score -= 16
            break
    if n.startswith(("ss", "ll", "rr")):
        score -= 30
    if any(s in n for s in ("asdf", "qwer", "zxcv", "ooo", "eee", "yyy")):
        score -= 50

    # Vowel skeleton diversity — kill a-a-a / e-e-e sludge
    v_seq = "".join(c for c in n if is_vowel(c))
    if len(set(v_seq)) == 1 and len(v_seq) >= 2:
        score -= 28
    if v_seq.count("a") >= 3:
        score -= 30
    if len(v_seq) >= 3 and len(set(v_seq)) < 2:
        score -= 20

    # Liquid overload (ralepler / raleler family)
    if n.count("r") >= 3 or n.count("l") >= 3:
        score -= 35
    if n.count("r") + n.count("l") >= 4:
        score -= 25

    uniq = len(set(n)) / len(n)
    if uniq < 0.5:
        score -= 22

    if n.endswith("y") and len(n) >= 6:
        score -= 8

    melodies = premium_melodies()
    if v_seq and v_seq not in melodies:
        close = any(
            abs(len(v_seq) - len(m)) <= 1
            and sum(a == b for a, b in zip(v_seq, m)) >= max(1, min(len(v_seq), len(m)) - 1)
            for m in melodies
        )
        score -= 8 if close else 22

    return max(0.0, min(100.0, score))


@lru_cache(maxsize=1)
def premium_melodies() -> frozenset[str]:
    return frozenset(
        "".join(c for c in w if is_vowel(c))
        for w in load_lines("brands_premium.txt")
        if 5 <= len(w) <= 9
    )


@lru_cache(maxsize=1)
def banned_frags() -> frozenset[str]:
    return frozenset(f for f in load_lines("banned_morphemes.txt") if len(f) >= 3)


@lru_cache(maxsize=1)
def english_cores() -> frozenset[str]:
    return frozenset(w for w in load_lines("english_words.txt") if len(w) >= 5)


def looks_premium_brand(name: str, min_score: float = 88.0, deep: bool = False) -> bool:
    """Fast structural gate. Set deep=True for full english/banned scan."""
    n = normalize(name)
    if brand_phonotactic_score(n) < min_score:
        return False
    v_seq = "".join(c for c in n if is_vowel(c))
    if v_seq.count("a") >= 3 or n.count("r") >= 3 or n.count("l") >= 3:
        return False
    bigs = brand_bigrams()
    if any(n[i : i + 2] not in bigs for i in range(len(n) - 1)):
        return False
    if len(n) >= 3:
        tris = brand_trigrams()
        attested = sum(1 for i in range(len(n) - 2) if n[i : i + 3] in tris)
        if attested / (len(n) - 2) < 0.55:
            return False
    if deep:
        if any(frag in n for frag in banned_frags()):
            return False
        if any(w in n for w in english_cores()):
            return False
    return True


def load_training_brands(min_len: int = 4, max_len: int = 12) -> list[str]:
    """Premium-only corpus for statistical generation (never emit verbatim)."""
    brands: list[str] = []
    source = load_lines("brands_premium.txt") or load_lines("brands_corpus.txt")
    for w in sorted(source):
        if min_len <= len(w) <= max_len:
            brands.append(w)
    return brands


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

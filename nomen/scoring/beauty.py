"""BeautyScore — independent aesthetic model trained on successful software brands.

Knows NOTHING about GitHub, domains, registries, companies, SEO, or collisions.
Learns only statistical orthographic / phonetic / visual properties.
Never memorizes names for emission.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache

from nomen.linguistics import (
    is_vowel,
    load_lines,
    max_consonant_run,
    max_vowel_run,
    normalize,
    onset_of,
)

# Patterns that scream "synthetic generator" — hard aesthetic rejects
_SYNTHETIC_FRAGS = (
    "gh",
    "qh",
    "yh",
    "phl",
    "gho",
    "ogh",
    "umin",
    "affive",
    "ogha",
    "logho",
    "yogh",
    "ffiv",
    "ublu",
    "ublin",
    "aogh",
    "eogh",
    "iogha",
    "xy",
    "yx",
    "qz",
    "zq",
    "vx",
    "xv",
    "kk",
    "vv",
    "ww",
    "jj",
)

# Fantasy / pharma / sci-fi residues that fail the premium test
_GENRE_FRAGS = (
    "wyn",
    "yth",
    "aeon",
    "xylo",
    "zyme",
    "cill",
    "mab",
    "olol",
    "pril",
    "xeno",
    "cyber",
    "neo",
    "ultra",
    "mega",
    "hyper",
    "omni",
)

_PREMIUM_OPENERS = frozenset("vfnmklpsrctdbhgjw")
_PREMIUM_ENDINGS = (
    "el",
    "en",
    "ay",
    "ey",
    "on",
    "an",
    "in",
    "io",
    "ia",
    "ix",
    "ex",
    "ly",
    "ry",
    "ty",
    "a",
    "o",
    "y",
    "e",
)
_UGLY_ENDINGS = frozenset("qjwxzucih")
_HARD_ONSETS = frozenset({"xh", "qh", "zx", "vx", "bx", "gx", "kx", "mx", "px", "tx", "wx"})
# Premium English onsets that look like 3-consonant runs but are natural (Stripe, Spline…)
_ALLOWED_CC_ONSETS = frozenset({"str", "spr", "scr", "spl", "thr", "chr", "sch", "shr"})
_AWKWARD_TAILS = (
    "cte",
    "pte",
    "kte",
    "ghe",
    "ecte",
    "orre",
    "arra",
    "erre",
    "ulle",
    "recte",
)


@dataclass(frozen=True)
class BeautyBreakdown:
    rhythm: float
    cadence: float
    phonotactics: float
    letter_balance: float
    visual: float
    consonant_flow: float
    vowel_spacing: float
    opening: float
    ending: float
    transitions: float
    readability: float
    memorability: float
    premium: float
    naturalness: float
    beauty_score: float
    reject_reasons: tuple[str, ...] = ()


def _syllables(name: str) -> list[str]:
    """Greedy CV+ syllable approximation (statistical, not linguistic truth)."""
    n = name
    out: list[str] = []
    i = 0
    while i < len(n):
        start = i
        while i < len(n) and not is_vowel(n[i]):
            i += 1
        while i < len(n) and is_vowel(n[i]):
            i += 1
        # trailing coda consonant if not start of next onset cluster needing vowel
        if i < len(n) and not is_vowel(n[i]):
            # take one coda unless next is also C then V (onset of next)
            if i + 1 < len(n) and is_vowel(n[i + 1]):
                pass
            else:
                i += 1
        if i == start:
            i += 1
        out.append(n[start:i])
    return [s for s in out if s]


def _approx_stress_pattern(name: str) -> str:
    """Crude stress: first syllable strong for short brands (Figma, Stripe, Notion)."""
    syl = _syllables(name)
    if not syl:
        return ""
    if len(syl) == 1:
        return "S"
    if len(syl) == 2:
        return "Sw"
    return "S" + "w" * (len(syl) - 1)


class BeautyModel:
    """Statistical aesthetic prior over successful software brand orthography."""

    def __init__(self) -> None:
        brands = self._load_brands()
        self.brands = brands
        self.n_brands = max(1, len(brands))
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[str] = Counter()
        self.trigrams: Counter[str] = Counter()
        self.starts1: Counter[str] = Counter()
        self.starts2: Counter[str] = Counter()
        self.ends1: Counter[str] = Counter()
        self.ends2: Counter[str] = Counter()
        self.ends3: Counter[str] = Counter()
        self.templates: Counter[str] = Counter()
        self.melodies: Counter[str] = Counter()
        self.stress: Counter[str] = Counter()
        self.syllable_counts: Counter[int] = Counter()
        self.lengths: Counter[int] = Counter()
        self.pos_unigram: dict[int, Counter[str]] = defaultdict(Counter)
        self._train(brands)
        # Smoothing baselines
        self._bi_total = sum(self.bigrams.values()) + 1
        self._tri_total = sum(self.trigrams.values()) + 1
        self._uni_total = sum(self.unigrams.values()) + 1

    @staticmethod
    def _load_brands() -> list[str]:
        # Prefer large corpus ∪ premium; patterns only, never emit verbatim.
        seen: set[str] = set()
        out: list[str] = []
        for src in ("brands_corpus.txt", "brands_premium.txt"):
            for w in load_lines(src):
                w = normalize(w)
                if 4 <= len(w) <= 12 and w.isalpha() and w not in seen:
                    seen.add(w)
                    out.append(w)
        return out

    def _train(self, brands: list[str]) -> None:
        for w in brands:
            self.lengths[len(w)] += 1
            self.unigrams.update(w)
            self.starts1[w[0]] += 1
            self.ends1[w[-1]] += 1
            if len(w) >= 2:
                self.starts2[w[:2]] += 1
                self.ends2[w[-2:]] += 1
            if len(w) >= 3:
                self.ends3[w[-3:]] += 1
            for i in range(len(w) - 1):
                self.bigrams[w[i : i + 2]] += 1
            for i in range(len(w) - 2):
                self.trigrams[w[i : i + 3]] += 1
            tmpl = "".join("V" if is_vowel(c) else "C" for c in w)
            self.templates[tmpl] += 1
            mel = "".join(c for c in w if is_vowel(c))
            if mel:
                self.melodies[mel] += 1
            syl = _syllables(w)
            self.syllable_counts[len(syl)] += 1
            self.stress[_approx_stress_pattern(w)] += 1
            for i, ch in enumerate(w):
                # relative position bucket 0..4
                bucket = min(4, int(i / max(len(w), 1) * 5))
                self.pos_unigram[bucket][ch] += 1

    def _log_bigram(self, bg: str) -> float:
        return math.log((self.bigrams[bg] + 0.5) / self._bi_total)

    def _log_trigram(self, tg: str) -> float:
        return math.log((self.trigrams[tg] + 0.25) / self._tri_total)

    def _pct(self, counter: Counter, key: str) -> float:
        tot = sum(counter.values()) or 1
        return counter[key] / tot

    def _has_awkward_cluster(self, n: str) -> bool:
        """True if there is a 3+ consonant run that is not a premium onset."""
        onset = onset_of(n)
        i = 0
        while i < len(n):
            if is_vowel(n[i]):
                i += 1
                continue
            j = i
            while j < len(n) and not is_vowel(n[j]):
                j += 1
            run = n[i:j]
            if len(run) >= 3:
                # Leading premium onset is fine (str-ipe)
                if i == 0 and run[:3] in _ALLOWED_CC_ONSETS and len(run) <= 3:
                    i = j
                    continue
                # Internal CCC almost always ugly in short brands
                return True
            i = j
        if onset in _HARD_ONSETS:
            return True
        return False

    def synthetic_rejects(self, name: str) -> list[str]:
        n = normalize(name)
        reasons: list[str] = []
        for frag in _SYNTHETIC_FRAGS:
            if frag in n:
                reasons.append(f"synthetic fragment '{frag}'")
        for frag in _GENRE_FRAGS:
            if frag in n and frag not in {"neo"}:  # neo appears in real brands occasionally
                reasons.append(f"genre fragment '{frag}'")
            elif frag == "neo" and n.startswith("neo") and len(n) > 5:
                reasons.append("genre fragment 'neo…'")
        for tail in _AWKWARD_TAILS:
            if n.endswith(tail) or tail in n:
                reasons.append(f"awkward tail/fragment '{tail}'")
                break
        # Doubled liquids that feel invented (horrecte) — rare in premium SaaS
        if "rr" in n or "ll" in n:
            # Allow only if digram is common in brand prior
            for dig in ("rr", "ll"):
                if dig in n and self.bigrams[dig] < max(3, self.n_brands * 0.01):
                    reasons.append(f"awkward double '{dig}'")
                    break
        # Machine-feel: too many rare letters
        rare = sum(n.count(c) for c in "qxzj")
        if rare >= 2:
            reasons.append("exotic letter pileup")
        if self._has_awkward_cluster(n):
            reasons.append("awkward consonant cluster")
        if max_vowel_run(n) >= 3:
            reasons.append("vowel pileup")
        onset = onset_of(n)
        if onset in _HARD_ONSETS:
            reasons.append(f"ugly onset '{onset}'")
        # Excessive entropy vs brand prior
        if len(n) >= 5:
            lp = sum(self._log_bigram(n[i : i + 2]) for i in range(len(n) - 1)) / (len(n) - 1)
            # brand mean bigram logprob roughly in [-8, -4]; below -10 is alien
            if lp < -9.5:
                reasons.append("excessive orthographic entropy")
        return reasons

    def score(self, name: str) -> BeautyBreakdown:
        n = normalize(name)
        rejects = tuple(self.synthetic_rejects(n))
        if not n or not n.isalpha():
            return BeautyBreakdown(
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, ("non-alpha",)
            )

        # --- rhythm / syllable cadence / stress ---
        syl = _syllables(n)
        n_syl = len(syl)
        syl_pref = self._pct(self.syllable_counts, n_syl)
        # Premium SaaS often 2 syllables (Fi-gma, No-tion, Ver-cel, Lin-ear≈2-3)
        rhythm = 55 + syl_pref * 120
        if n_syl in (2, 3):
            rhythm += 18
        elif n_syl == 1 and 5 <= len(n) <= 6:
            rhythm += 12
        else:
            rhythm -= 15
        # Even syllable lengths feel cleaner
        if syl and max(len(s) for s in syl) <= 4:
            rhythm += 8
        stress = _approx_stress_pattern(n)
        cadence = 60 + self._pct(self.stress, stress) * 100
        if stress in ("Sw", "S", "Sww"):
            cadence += 12

        # --- phonotactics via brand n-grams (learned, not memorized names) ---
        bi_lp = [self._log_bigram(n[i : i + 2]) for i in range(len(n) - 1)]
        tri_lp = [self._log_trigram(n[i : i + 3]) for i in range(len(n) - 2)] if len(n) >= 3 else [0.0]
        mean_bi = sum(bi_lp) / max(len(bi_lp), 1)
        mean_tri = sum(tri_lp) / max(len(tri_lp), 1)
        # Map logprob into 0–100
        phonotactics = max(0.0, min(100.0, 100 + (mean_bi + 6.0) * 18 + (mean_tri + 8.0) * 6))

        # --- letter balance vs brand unigram prior ---
        kl = 0.0
        for ch, p in self.unigrams.items():
            q = n.count(ch) / len(n)
            p_norm = p / self._uni_total
            if q > 0:
                kl += q * math.log((q + 1e-9) / (p_norm + 1e-9))
        letter_balance = max(0.0, min(100.0, 92 - kl * 55))
        vowels = sum(1 for c in n if is_vowel(c))
        v_ratio = vowels / len(n)
        if 0.35 <= v_ratio <= 0.55:
            letter_balance += 8
        else:
            letter_balance -= 12

        # --- visual symmetry / identity ---
        asc = set("bdfhiklt")
        desc = set("gjpqy")
        a = sum(1 for c in n if c in asc)
        d = sum(1 for c in n if c in desc)
        mid = len(n) - a - d
        visual = 62 + mid * 4
        if a and d:
            visual += 10
        if a + d <= len(n) * 0.55:
            visual += 6
        # Double letters only if brand-like (ll, ss rare ok)
        for i in range(len(n) - 1):
            if n[i] == n[i + 1] and n[i] not in "lnrs":
                visual -= 18
                break
        # Compact length = strong logo mark
        if 5 <= len(n) <= 7:
            visual += 12
        elif len(n) == 8:
            visual += 4
        else:
            visual -= 8

        # --- consonant flow ---
        cr = max_consonant_run(n)
        consonant_flow = 88.0
        if self._has_awkward_cluster(n):
            consonant_flow -= 40
        elif cr >= 3 and onset_of(n)[:3] in _ALLOWED_CC_ONSETS:
            consonant_flow += 4  # Stripe-like onset is a feature
        elif cr == 2:
            # Allowed liquids/stop clusters common in brands (str, cl, br)
            ok_clusters = 0
            for i in range(len(n) - 1):
                if not is_vowel(n[i]) and not is_vowel(n[i + 1]):
                    bg = n[i : i + 2]
                    if self.bigrams[bg] >= 3:
                        ok_clusters += 1
                    else:
                        consonant_flow -= 14
            consonant_flow += min(10, ok_clusters * 3)
        soft = sum(n.count(c) for c in "rlnm")
        if soft >= 4:
            consonant_flow -= 20

        # --- vowel spacing ---
        vowel_spacing = 80.0
        vr = max_vowel_run(n)
        if vr >= 3:
            vowel_spacing -= 35
        elif vr == 2:
            vowel_spacing -= 8
        # Alternation bonus
        alt = sum(1 for i in range(len(n) - 1) if is_vowel(n[i]) != is_vowel(n[i + 1]))
        vowel_spacing += min(15, alt * 2.5)
        mel = "".join(c for c in n if is_vowel(c))
        if mel and self.melodies[mel] > 0:
            vowel_spacing += min(15, 4 + math.log1p(self.melodies[mel]) * 3)
        elif mel and len(set(mel)) == 1:
            vowel_spacing -= 25

        # --- opening quality ---
        opening = 55 + self._pct(self.starts1, n[0]) * 160
        if n[0] in _PREMIUM_OPENERS:
            opening += 10
        if n[0] in "qxz":
            opening -= 25
        if len(n) >= 2:
            opening += self._pct(self.starts2, n[:2]) * 80

        # --- ending quality ---
        ending = 50.0
        if any(n.endswith(e) for e in _PREMIUM_ENDINGS):
            ending += 22
        if n[-1] in _UGLY_ENDINGS:
            ending -= 28
        ending += self._pct(self.ends1, n[-1]) * 60
        if len(n) >= 2:
            ending += self._pct(self.ends2, n[-2:]) * 90
        if len(n) >= 3:
            ending += min(15, self._pct(self.ends3, n[-3:]) * 120)

        # --- internal transitions ---
        transitions = phonotactics * 0.7
        tmpl = "".join("V" if is_vowel(c) else "C" for c in n)
        transitions += self._pct(self.templates, tmpl) * 80
        # Positional letter fit
        pos_fit = 0.0
        for i, ch in enumerate(n):
            bucket = min(4, int(i / max(len(n), 1) * 5))
            pos_fit += self._pct(self.pos_unigram[bucket], ch)
        transitions += min(20, pos_fit * 8)

        # --- readability (pronounce without hearing) ---
        readability = 70.0
        readability += min(15, alt * 2)
        if cr <= 2 and vr <= 2:
            readability += 12
        if rejects:
            readability -= 30
        # Silent-letter traps
        for trap in ("gh", "kh", "rh", "mn", "bt", "pt"):
            if trap in n:
                readability -= 18

        # --- memorability (short footprint, once-read) ---
        memorability = 60.0
        if 5 <= len(n) <= 7:
            memorability += 22
        elif len(n) == 8:
            memorability += 10
        else:
            memorability -= 10
        memorability += min(12, len(set(n)) * 1.5)
        if n_syl <= 3:
            memorability += 10
        if uniq := len(set(n)) / len(n):
            if uniq < 0.45:
                memorability -= 20

        # --- premium / billion-dollar homepage test ---
        # Target feel: Figma, Stripe, Notion, Vercel, Linear — punchy, not floaty
        premium = 58.0
        if 5 <= len(n) <= 7:
            premium += 14
        elif len(n) >= 9:
            premium -= 10
        if n_syl == 2:
            premium += 14
        elif n_syl == 3:
            premium += 6
        else:
            premium -= 8
        # Confident endings (Vercel, Linear, Notion, Stripe, Figma, Framer)
        if any(
            n.endswith(e)
            for e in ("el", "ay", "ey", "on", "an", "in", "ar", "er", "or", "ex", "ix", "e", "y")
        ):
            premium += 14
        elif n.endswith(("a", "o", "io", "ia")):
            premium += 2  # allowed but not preferred vs punchy tails
        else:
            premium -= 10
        # Minimal / confident: not too ornate
        if sum(1 for c in n if c in "qxzj") == 0:
            premium += 8
        soft_ratio = soft / len(n)
        # Liquid sludge — but don't punish Linear/Klarna-like shapes with strong priors
        if soft_ratio > 0.45 and mean_bi < -5.8:
            premium -= 18
        elif soft_ratio > 0.55:
            premium -= 12
        if soft >= 3 and n.count("a") >= 2 and n[-1] in "aoe":
            premium -= 20  # fantasy sludge
        # Pure CVCVCV Romance float (…ata/…ora/…iva) — game/fantasy, not SaaS
        if (
            tmpl.startswith("CVCV")
            and n.endswith(("a", "o"))
            and soft_ratio >= 0.30
            and mean_bi < -5.5
        ):
            premium -= 16
        if n.count("a") >= 2 and n[-1] in "ao":
            premium -= 12
        if re.search(r"(.)\1\1", n):
            premium -= 25
        # Stop/fricative presence = visual bite (Stripe, Figma, Click…)
        hard = sum(1 for c in n if c in "bdfgkptvzxcs")
        if hard >= 2:
            premium += 8
        elif hard == 0:
            premium -= 14

        # --- naturalness (anti machine-generated) ---
        naturalness = 75.0 + (mean_bi + 6.5) * 10
        if rejects:
            naturalness -= 40
        # Unique letter ratio too high with rare letters = invented soup
        if sum(1 for c in n if self.unigrams[c] < self.n_brands * 0.02) >= 2:
            naturalness -= 18
        if soft_ratio > 0.5 and mean_bi < -5.8:
            naturalness -= 15
        if tmpl.count("V") >= len(tmpl) * 0.5 and n.endswith("a") and mean_bi < -5.5:
            naturalness -= 12

        def clamp(x: float) -> float:
            return max(0.0, min(100.0, x))

        comps = {
            "rhythm": clamp(rhythm),
            "cadence": clamp(cadence),
            "phonotactics": clamp(phonotactics),
            "letter_balance": clamp(letter_balance),
            "visual": clamp(visual),
            "consonant_flow": clamp(consonant_flow),
            "vowel_spacing": clamp(vowel_spacing),
            "opening": clamp(opening),
            "ending": clamp(ending),
            "transitions": clamp(transitions),
            "readability": clamp(readability),
            "memorability": clamp(memorability),
            "premium": clamp(premium),
            "naturalness": clamp(naturalness),
        }

        beauty = (
            comps["rhythm"] * 0.07
            + comps["cadence"] * 0.06
            + comps["phonotactics"] * 0.12
            + comps["letter_balance"] * 0.06
            + comps["visual"] * 0.08
            + comps["consonant_flow"] * 0.08
            + comps["vowel_spacing"] * 0.07
            + comps["opening"] * 0.07
            + comps["ending"] * 0.09
            + comps["transitions"] * 0.07
            + comps["readability"] * 0.08
            + comps["memorability"] * 0.07
            + comps["premium"] * 0.10
            + comps["naturalness"] * 0.08
        )

        # Hard fails collapse the score
        if rejects:
            beauty = min(beauty, 42.0) - 8 * len(rejects)

        # Homepage belief gate: if naturalness or premium tank, cap
        if comps["premium"] < 60 or comps["naturalness"] < 55:
            beauty = min(beauty, 68.0)
        if comps["premium"] < 55 or comps["naturalness"] < 50:
            beauty = min(beauty, 58.0)
        if comps["readability"] < 55 or comps["memorability"] < 55:
            beauty = min(beauty, 62.0)

        return BeautyBreakdown(
            rhythm=comps["rhythm"],
            cadence=comps["cadence"],
            phonotactics=comps["phonotactics"],
            letter_balance=comps["letter_balance"],
            visual=comps["visual"],
            consonant_flow=comps["consonant_flow"],
            vowel_spacing=comps["vowel_spacing"],
            opening=comps["opening"],
            ending=comps["ending"],
            transitions=comps["transitions"],
            readability=comps["readability"],
            memorability=comps["memorability"],
            premium=comps["premium"],
            naturalness=comps["naturalness"],
            beauty_score=round(clamp(beauty), 2),
            reject_reasons=rejects,
        )


@lru_cache(maxsize=1)
def get_beauty_model() -> BeautyModel:
    return BeautyModel()


def beauty_score(name: str) -> float:
    return get_beauty_model().score(name).beauty_score


def beauty_breakdown(name: str) -> BeautyBreakdown:
    return get_beauty_model().score(name)


def passes_beauty_gates(name: str, min_beauty: float = 72.0) -> tuple[bool, list[str]]:
    """Visual + readability + memory + premium tests."""
    bd = beauty_breakdown(name)
    reasons: list[str] = list(bd.reject_reasons)
    if bd.beauty_score < min_beauty:
        reasons.append(f"beauty {bd.beauty_score:.1f} < {min_beauty}")
    if bd.readability < 60:
        reasons.append(f"readability {bd.readability:.1f}")
    if bd.memorability < 60:
        reasons.append(f"memorability {bd.memorability:.1f}")
    if bd.premium < 58:
        reasons.append(f"premium {bd.premium:.1f}")
    if bd.naturalness < 55:
        reasons.append(f"feels generated ({bd.naturalness:.1f})")
    return (not reasons), reasons

"""Aggressive offline rejection bank."""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein

from nomen.linguistics import (
    brand_phonotactic_score,
    english_cores,
    load_lines,
    normalize,
    occupied_brands,
    reserved_brands,
)
from nomen.models import Candidate, Stage

_WEAK_TAILS = (
    "acker", "aster", "ester", "ender", "eller", "erer", "aler", "oter",
    "ater", "iter", "uter", "eker", "iker", "oker", "uker", "ster", "fter",
    "nter", "rker", "yter",
)
_OVERUSED_PREFIXES = ("ser", "ter", "mar", "per", "ste", "con", "pla", "pro", "cor")

# Synthetic-generator tells that kill premium brand feel
_SYNTHETIC_FRAGS = (
    "gh", "qh", "yh", "phl", "gho", "ogh", "umin", "affive",
    "ogha", "logho", "yogh", "ffiv", "ublu", "ublin",
    "ecte", "orre", "recte",
)


class OfflineFilterBank:
    def __init__(self) -> None:
        self.banned = load_lines("banned_morphemes.txt")
        self.english = {w for w in load_lines("english_words.txt") if 4 <= len(w) <= 12}
        self.cores = english_cores()
        self.known = set(occupied_brands())
        self.reserved = set(reserved_brands())
        self._english_by_len: dict[int, list[str]] = {}
        for w in self.english:
            self._english_by_len.setdefault(len(w), []).append(w)

    def check(self, name: str) -> list[str]:
        """Return rejection reasons (empty if ok)."""
        n = normalize(name)
        reasons: list[str] = []

        score = brand_phonotactic_score(n)
        if score < 88:
            reasons.append(f"quality: phonotactic {score:.1f} < 88")

        for frag in _SYNTHETIC_FRAGS:
            if frag in n:
                reasons.append(f"quality: synthetic fragment '{frag}'")
                break

        if n in self.reserved:
            reasons.append("reserved: occupied brand")
        elif n in self.known:
            reasons.append("banned: verbatim training brand")

        for frag in self.banned:
            if frag and frag in n:
                reasons.append(f"banned: contains '{frag}'")
                break

        if any(n.endswith(t) for t in _WEAK_TAILS):
            reasons.append("quality: weak English-like agentive/tail pattern")
        if n.startswith(_OVERUSED_PREFIXES) and n.endswith(("er", "ar", "or", "a", "o")):
            if brand_phonotactic_score(n) < 96:
                reasons.append("quality: overused prefix/tail brand sludge")
        if n.endswith(("era", "ara", "ery", "ory", "ina", "ana", "elo", "alo")):
            if n.count("r") + n.count("l") >= 2:
                reasons.append("quality: liquid–vowel sludge ending")
        soft = sum(n.count(c) for c in "rln")
        if soft >= 4:
            reasons.append("quality: too many soft consonants (r/l/n)")

        if n in self.english:
            reasons.append(f"english: exact word '{n}'")
        else:
            hit = next((w for w in self.cores if w in n), None)
            if hit:
                reasons.append(f"english: contains '{hit}'")
            else:
                for w in self.english:
                    if len(w) < 4:
                        continue
                    if n.startswith(w) or n.endswith(w):
                        residue = n[len(w) :] if n.startswith(w) else n[: -len(w)]
                        if residue in {"", "s", "es", "ed", "er", "ing", "y", "ly"}:
                            reasons.append(f"english: contains '{w}'")
                            break

            for other in self._english_by_len.get(len(n), []):
                if Levenshtein.distance(n, other) <= 1:
                    reasons.append(f"english: near-word '{other}'")
                    break

        return reasons

    def apply(self, provenance: dict[str, str]) -> tuple[list[Candidate], list[Candidate]]:
        ok: list[Candidate] = []
        bad: list[Candidate] = []
        for name, gen in provenance.items():
            cand = Candidate(name=name, generator=gen)
            reasons = self.check(name)
            if reasons:
                for r in reasons:
                    stage = Stage.BANNED
                    if r.startswith("english"):
                        stage = Stage.ENGLISH
                    elif r.startswith("quality"):
                        stage = Stage.QUALITY
                    cand.reject(stage, r.split(": ", 1)[-1])
                bad.append(cand)
            else:
                ok.append(cand)
        return ok, bad

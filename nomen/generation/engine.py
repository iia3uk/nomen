"""
Novelty-first generation engine.

Diversity is a hard constraint. Generators run under quotas.
Evolution injects: 30% random · 20% crossover · 20% mutation · 30% fresh generators.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable

from nomen.diversity.features import cv_pattern, soft_consonant_ratio, vowel_a_ratio
from nomen.hashing import stable_seed
from nomen.linguistics import brand_phonotactic_score, is_vowel, looks_premium_brand, normalize
from nomen.training.models import BrandLanguageModels

# Hard generator mix for the merged batch (no generator > 20% of final —
# enforced later; generation quotas keep raw supply balanced).
GENERATOR_QUOTAS: dict[str, float] = {
    "transformer": 0.15,
    "evolutionary": 0.15,
    "genetic": 0.10,
    "phoneme": 0.15,
    "transition_graph": 0.10,
    "entropy": 0.10,
    "char_lm": 0.10,
    "experimental": 0.15,
}

LENGTH_WEIGHTS = {5: 15, 6: 25, 7: 30, 8: 20, 9: 10}
# Split each generator quota so idle workers pick up remaining chunks.
_GEN_CHUNK = 1200

# Onsets / nuclei / codas drawn from premium brand statistics (not fixed dictionary words)
_ONSETS = (
    "b", "br", "bl", "c", "cl", "cr", "d", "dr", "f", "fl", "fr", "g", "gl", "gr",
    "h", "j", "k", "kl", "kr", "m", "n", "p", "pl", "pr", "qu", "r", "s", "sc",
    "sk", "sl", "sm", "sn", "sp", "st", "t", "tr", "v", "w", "z",
)
_NUCLEI = ("a", "e", "i", "o", "u", "ai", "au", "ea", "io", "oa", "oi")
_CODAS = ("", "", "n", "r", "s", "l", "m", "x", "v", "st", "nd", "nt", "rk", "lt")


class GenerationEngine:
    def __init__(
        self,
        seed: str,
        min_len: int = 5,
        max_len: int = 9,
        workers: int | None = None,
    ) -> None:
        self.min_len = min_len
        self.max_len = max_len
        self.workers = workers
        self.base_seed = stable_seed(seed, bits=31)
        self.exploration_salt = 0
        self.seed_str = seed
        self.lm = BrandLanguageModels(seed=self.base_seed)
        self.rng = random.Random(self.base_seed)
        self.archive: list[str] = []  # diverse behavioral archive (not just winners)
        self.winners: list[str] = []
        self._brand_set = set(self.lm.brands)
        self._letter_pressure: Counter[str] = Counter()
        self._ending_pressure: Counter[str] = Counter()
        self._ending_cap = 24
        self.last_stats: dict[str, int] = {}

    def reseed_exploration(self, reason: str = "") -> None:
        """Jump to a new random region of search space (anti-convergence)."""
        self.exploration_salt += 1
        self.rng = random.Random(self.base_seed ^ (self.exploration_salt * 0x9E3779B9))
        self.lm = BrandLanguageModels(seed=self.base_seed + self.exploration_salt * 9973)
        self._letter_pressure.clear()
        self._ending_pressure.clear()
        # Keep winners, wipe local mutation archive to escape basin
        self.archive = list(self.winners)

    def set_winners(self, names: list[str]) -> None:
        self.winners = [normalize(n) for n in names if n]

    def set_archive(self, names: list[str]) -> None:
        self.archive = [normalize(n) for n in names if n]

    def _sample_length(self) -> int:
        lengths = [L for L in range(self.min_len, self.max_len + 1)]
        weights = [LENGTH_WEIGHTS.get(L, 5) for L in lengths]
        return self.rng.choices(lengths, weights=weights, k=1)[0]

    def _accept(self, name: str) -> bool:
        n = normalize(name)
        if not (self.min_len <= len(n) <= self.max_len):
            return False
        if n in self._brand_set or n in self.winners:
            return False
        if soft_consonant_ratio(n) > 0.50:
            return False
        if vowel_a_ratio(n) > 0.35:
            return False
        if n.count("r") >= 3 or n.count("l") >= 3:
            return False
        if brand_phonotactic_score(n) < 72:
            return False
        # Letter-pressure: reject if batch already flooded with same letters
        pressure = sum(self._letter_pressure[c] for c in set(n) & set("rlnsa"))
        if pressure > 40 and soft_consonant_ratio(n) > 0.35:
            return False
        end = n[-2:] if len(n) >= 2 else n
        if self._ending_pressure[end] >= self._ending_cap:
            return False
        return True

    def _commit(self, name: str) -> None:
        for c in name:
            self._letter_pressure[c] += 1
        if len(name) >= 2:
            self._ending_pressure[name[-2:]] += 1

    def _sample_from_probs(self, probs: dict[str, float], temperature: float = 1.0) -> str:
        items = []
        for c, p in probs.items():
            # Down-weight overused soft letters
            pen = 1.0
            if c in "rlns" and self._letter_pressure[c] > 12:
                pen = 0.35
            if c == "a" and self._letter_pressure["a"] > 20:
                pen = 0.4
            items.append((c, max(p, 1e-12) ** (1 / temperature) * pen))
        letters, weights = zip(*items)
        return self.rng.choices(list(letters), weights=list(weights), k=1)[0]

    # ---- Generators ---------------------------------------------------------

    def gen_transformer(self, count: int) -> list[str]:
        out: set[str] = set()
        attempts = 0
        limit = max(count * 30, 400)
        while len(out) < count and attempts < limit:
            attempts += 1
            length = self._sample_length()
            starts = {c: float(self.lm.starts[c] + 1) for c in "bcdfghjkmnptvwz"}
            chars = [self._sample_from_probs(starts, 1.0)]
            for _ in range(length - 1):
                probs = self.lm.transformer_next_probs("".join(chars))
                chars.append(self._sample_from_probs(probs, 1.05))
            name = "".join(chars)
            if self._accept(name):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_char_lm(self, count: int) -> list[str]:
        out: set[str] = set()
        templates = [t for t, c in self.lm.templates.most_common(60) if c >= 1]
        if not templates:
            templates = ["CVCVC", "CCVCV", "CVCCVC", "CVCVCV", "VCVCVC", "CVVCVC"]
        # Force pattern diversity in sampling
        attempts = 0
        limit = max(count * 35, 500)
        used_patterns: Counter[str] = Counter()
        while len(out) < count and attempts < limit:
            attempts += 1
            tmpl = self.rng.choice(templates)
            if used_patterns[tmpl] > count * 0.25 + 2:
                continue
            chars: list[str] = []
            for i, slot in enumerate(tmpl):
                dist = self.lm.pos_chars.get((tmpl, i)) or Counter(
                    "aeiou" if slot == "V" else "bcdfghkmnptvwz"
                )
                if chars:
                    trans = self.lm.next_given.get(chars[-1], Counter())
                    blended: Counter[str] = Counter()
                    for ch, w in dist.items():
                        ok = (slot == "V" and is_vowel(ch)) or (slot == "C" and not is_vowel(ch))
                        if ok:
                            blended[ch] = w + trans.get(ch, 0)
                    dist = blended or dist
                letters = list(dist.keys())
                weights = [float(dist[c]) for c in letters]
                # Soft-letter penalty
                weights = [
                    w * (0.3 if (letters[i] in "rlns" and self._letter_pressure[letters[i]] > 10) else 1.0)
                    for i, w in enumerate(weights)
                ]
                chars.append(self.rng.choices(letters, weights=weights, k=1)[0])
            name = "".join(chars)
            if self.min_len <= len(name) <= self.max_len and self._accept(name):
                out.add(name)
                used_patterns[cv_pattern(name)] += 1
                self._commit(name)
        return sorted(out)

    def gen_phoneme(self, count: int) -> list[str]:
        out: set[str] = set()
        attempts = 0
        limit = max(count * 30, 400)
        while len(out) < count and attempts < limit:
            attempts += 1
            length = self._sample_length()
            state = "^"
            letters: list[str] = []
            for _ in range(length + 4):
                dist = self.lm.phoneme_trans.get(state)
                if not dist:
                    break
                keys = list(dist.keys())
                weights = [float(dist[k]) for k in keys]
                nxt = self.rng.choices(keys, weights=weights, k=1)[0]
                if nxt == "$":
                    break
                letters.append(nxt[-1])
                state = nxt
                if len(letters) >= length:
                    break
            name = "".join(letters)
            if self._accept(name):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_transition_graph(self, count: int) -> list[str]:
        out: set[str] = set()
        attempts = 0
        limit = max(count * 30, 400)
        # Prefer less-used starts
        start_pool = [c for c, _ in self.lm.starts.most_common(30)]
        start_pool = [c for c in start_pool if c not in "rlns"] + [c for c in start_pool if c in "rlns"]
        while len(out) < count and attempts < limit:
            attempts += 1
            length = self._sample_length()
            node = self.rng.choice(start_pool or list("bcdfghkmptvwz"))
            path = [node]
            for _ in range(length - 1):
                edges = self.lm.next_given.get(path[-1])
                if not edges:
                    break
                top = edges.most_common(12)
                letters = [x[0] for x in top]
                weights = [
                    float(x[1]) * (0.25 if x[0] in "rlns" and self._letter_pressure[x[0]] > 10 else 1.0)
                    for x in top
                ]
                path.append(self.rng.choices(letters, weights=weights, k=1)[0])
            name = "".join(path)
            if self._accept(name):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_entropy(self, count: int) -> list[str]:
        out: set[str] = set()
        attempts = 0
        limit = max(count * 30, 400)
        while len(out) < count and attempts < limit:
            attempts += 1
            length = self._sample_length()
            chars = [self.rng.choice(list("bcdfghkmptvwz"))]
            for _ in range(length - 1):
                dist = self.lm.next_given.get(chars[-1]) or self.lm.unigrams
                total = sum(dist.values()) or 1
                items = [
                    (ch, (cnt / total) ** 0.9 * (0.3 if ch in "rlns" else 1.0))
                    for ch, cnt in dist.items()
                ]
                letters, weights = zip(*items)
                chars.append(self.rng.choices(list(letters), weights=list(weights), k=1)[0])
            name = "".join(chars)
            if self._accept(name):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_experimental(self, count: int) -> list[str]:
        """High-temperature / exotic structure explorer — maximizes novelty."""
        out: set[str] = set()
        attempts = 0
        limit = max(count * 40, 600)
        exotic_onsets = ("qu", "z", "v", "w", "j", "x", "sk", "sp", "tw", "dw", "gw", "kv")
        while len(out) < count and attempts < limit:
            attempts += 1
            mode = self.rng.random()
            if mode < 0.34:
                # Structured assemble with exotic onset
                name = normalize(
                    self.rng.choice(exotic_onsets)
                    + self.rng.choice(_NUCLEI)
                    + self.rng.choice(("b", "d", "g", "k", "m", "p", "t", "v", "z"))
                    + self.rng.choice(_NUCLEI)
                    + self.rng.choice(_CODAS)
                )
            elif mode < 0.67:
                # CVVCVC / VCVCVC templates
                tmpl = self.rng.choice(["CVVCVC", "VCVCVC", "CCVCVC", "CVCVVC", "CVCCVC"])
                chars: list[str] = []
                for slot in tmpl:
                    if slot == "V":
                        chars.append(self.rng.choice(list("aeiou")))
                    else:
                        chars.append(self.rng.choice(list("bcdfghkmptvwz")))
                name = "".join(chars)
            else:
                # High-temperature LM walk
                length = self._sample_length()
                chars = [self.rng.choice(list("bcdfghjkmpqtvwz"))]
                for _ in range(length - 1):
                    probs = self.lm.transformer_next_probs("".join(chars))
                    chars.append(self._sample_from_probs(probs, temperature=1.45))
                name = "".join(chars)
            if self._accept(name) and looks_premium_brand(name, min_score=70.0, deep=False):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_random_inject(self, count: int) -> list[str]:
        """Completely fresh random individuals (30% evolution slot)."""
        return self.gen_experimental(count)

    def gen_crossover(self, count: int) -> list[str]:
        """Crossover across diverse archive — not winner-only."""
        pool = list(dict.fromkeys(self.archive + self.winners + self.lm.brands[:80]))
        if len(pool) < 4:
            pool = list(self.lm.brands[:40]) or ["figma", "vercel", "notion", "stripe"]
        out: set[str] = set()
        attempts = 0
        limit = max(count * 40, 500)
        while len(out) < count and attempts < limit:
            attempts += 1
            a, b = self.rng.sample(pool, 2)
            # Multi-point / alternating crossover for diversity
            if self.rng.random() < 0.5:
                cut = self.rng.randint(1, min(len(a), len(b)) - 1)
                child = normalize(a[:cut] + b[cut:])
            else:
                child = normalize("".join(
                    (a[i] if i < len(a) else b[i % len(b)])
                    if i % 2 == 0
                    else (b[i] if i < len(b) else a[i % len(a)])
                    for i in range(max(len(a), len(b)))
                ))
            # Trim/pad to length target
            target = self._sample_length()
            if len(child) > target:
                child = child[:target]
            elif len(child) < target:
                child = (child + self.rng.choice(list("aeioumnptk")))[:target]
            if self._accept(child):
                out.add(child)
                self._commit(child)
        return sorted(out)

    def gen_mutation(self, count: int) -> list[str]:
        """Mutate diverse archive members (not only winners)."""
        pool = list(dict.fromkeys(self.archive + self.lm.brands[:60]))
        if not pool:
            pool = ["figma", "linear", "raycast", "stripe"]
        out: set[str] = set()
        attempts = 0
        limit = max(count * 35, 400)
        while len(out) < count and attempts < limit:
            attempts += 1
            base = list(normalize(self.rng.choice(pool)))
            ops = self.rng.randint(2, 5)
            for _ in range(ops):
                if not base:
                    break
                op = self.rng.choice(("sub", "ins", "del", "swap", "onset", "vowel_shift"))
                if op == "sub":
                    i = self.rng.randrange(len(base))
                    alphabet = "aeiou" if is_vowel(base[i]) else "bcdfghkmptvwz"
                    base[i] = self.rng.choice(alphabet)
                elif op == "ins" and len(base) < self.max_len:
                    i = self.rng.randrange(len(base) + 1)
                    base.insert(i, self.rng.choice("aeioubcdfghkmptvwz"))
                elif op == "del" and len(base) > self.min_len:
                    del base[self.rng.randrange(len(base))]
                elif op == "swap" and len(base) > 2:
                    i = self.rng.randrange(len(base) - 1)
                    base[i], base[i + 1] = base[i + 1], base[i]
                elif op == "onset":
                    base[0] = self.rng.choice(list("bcdfghjkmpqtvwz"))
                elif op == "vowel_shift":
                    idxs = [i for i, c in enumerate(base) if is_vowel(c)]
                    if idxs:
                        i = self.rng.choice(idxs)
                        base[i] = self.rng.choice([v for v in "aeiou" if v != base[i]] or list("aeiou"))
            name = normalize("".join(base))
            if self._accept(name):
                out.add(name)
                self._commit(name)
        return sorted(out)

    def gen_evolutionary(self, count: int) -> list[str]:
        """
        Evolution generation with mandatory injection mix:
        30% random · 20% crossover · 20% mutation · 30% fresh generators.
        """
        n_random = int(count * 0.30)
        n_cross = int(count * 0.20)
        n_mut = int(count * 0.20)
        n_fresh = count - n_random - n_cross - n_mut
        parts: list[str] = []
        parts += self.gen_random_inject(n_random)
        parts += self.gen_crossover(n_cross)
        parts += self.gen_mutation(n_mut)
        # Fresh = mix of independent generators
        fresh_each = max(1, n_fresh // 4)
        parts += self.gen_transformer(fresh_each)
        parts += self.gen_phoneme(fresh_each)
        parts += self.gen_entropy(fresh_each)
        parts += self.gen_experimental(n_fresh - 3 * fresh_each)
        # Dedup preserve order
        seen: set[str] = set()
        out: list[str] = []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:count]

    def gen_genetic(self, count: int) -> list[str]:
        return self.gen_mutation(count)

    def _snapshot_pressure(self) -> tuple[dict[str, int], dict[str, int]]:
        """Letter/ending counts from archive + winners, shared into worker processes."""
        letters: Counter[str] = Counter()
        endings: Counter[str] = Counter()
        for raw in list(self.archive[-200:]) + list(self.winners[-64:]):
            n = normalize(raw)
            letters.update(n)
            if len(n) >= 2:
                endings[n[-2:]] += 1
        return dict(letters), dict(endings)

    def _cull_saturated_endings(
        self,
        provenance: dict[str, str],
        *,
        max_ending_share: float = 0.06,
    ) -> dict[str, str]:
        """Drop cross-generator ending collapse after isolated workers merge."""
        names = list(provenance.keys())
        if len(names) < 40:
            return provenance
        cap = max(8, int(len(names) * max_ending_share))
        order = names[:]
        self.rng.shuffle(order)
        used: Counter[str] = Counter()
        kept: dict[str, str] = {}
        for name in order:
            end = name[-2:] if len(name) >= 2 else name
            if used[end] >= cap:
                continue
            used[end] += 1
            kept[name] = provenance[name]
        return kept

    def generate_all(
        self,
        total: int,
        on_generator: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, str]:
        """Merge generators under quotas — parallel across CPU cores."""
        from nomen.parallel import parallel_generate

        letters, endings = self._snapshot_pressure()
        self._letter_pressure = Counter(letters)
        self._ending_pressure = Counter(endings)
        shares = {k: max(1, int(total * v)) for k, v in GENERATOR_QUOTAS.items()}
        while sum(shares.values()) < total:
            shares["experimental"] += 1
        while sum(shares.values()) > total:
            for k in shares:
                if shares[k] > 1:
                    shares[k] -= 1
                    break

        payloads: list[dict[str, object]] = []
        for label, count in shares.items():
            left = count
            chunk_i = 0
            while left > 0:
                n = min(_GEN_CHUNK, left)
                payloads.append(
                    {
                        "label": label,
                        "count": n,
                        "chunk": chunk_i,
                        "seed": self.seed_str,
                        "salt": self.exploration_salt,
                        "min_len": self.min_len,
                        "max_len": self.max_len,
                        "archive": self.archive[-200:],
                        "winners": self.winners[-64:],
                        "pressure": letters,
                        "endings": endings,
                    }
                )
                left -= n
                chunk_i += 1
        provenance = parallel_generate(payloads, workers=self.workers, on_done=on_generator)
        before = len(provenance)
        provenance = self._cull_saturated_endings(provenance)
        self.last_stats = {
            "raw_unique": before,
            "kept": len(provenance),
            "culled": before - len(provenance),
            "engines": len(payloads),
        }

        sample = list(provenance.keys())
        self.rng.shuffle(sample)
        self.archive = list(dict.fromkeys(self.archive + sample[:200]))[-800:]
        self._letter_pressure.clear()
        self._ending_pressure.clear()
        for name in provenance:
            self._commit(name)
        return provenance

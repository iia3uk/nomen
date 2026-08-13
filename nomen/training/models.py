"""Learned language models over brand orthography (patterns only, never copy)."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

import numpy as np

from nomen.hashing import stable_seed
from nomen.linguistics import is_vowel, load_training_brands, normalize


class BrandLanguageModels:
    """Shared statistical structure extracted from thousands of software brands."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.brands = load_training_brands()
        self.unigrams: Counter[str] = Counter()
        self.bigrams: Counter[str] = Counter()
        self.trigrams: Counter[str] = Counter()
        self.starts: Counter[str] = Counter()
        self.ends: Counter[str] = Counter()
        self.next_given: dict[str, Counter[str]] = defaultdict(Counter)
        self.next2_given: dict[str, Counter[str]] = defaultdict(Counter)
        self.length_hist: Counter[int] = Counter()
        self.syllable_mass: Counter[str] = Counter()
        self.phoneme_trans: dict[str, Counter[str]] = defaultdict(Counter)
        self.templates: Counter[str] = Counter()
        self.pos_chars: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        self._train()
        self._build_transformer_weights()

    def _train(self) -> None:
        for w in self.brands:
            w = normalize(w)
            if len(w) < 3:
                continue
            self.length_hist[len(w)] += 1
            self.starts[w[0]] += 1
            self.ends[w[-2:] if len(w) >= 2 else w[-1]] += 1
            self.unigrams.update(w)
            for i in range(len(w) - 1):
                bg = w[i : i + 2]
                self.bigrams[bg] += 1
                self.next_given[w[i]][w[i + 1]] += 1
            for i in range(len(w) - 2):
                tg = w[i : i + 3]
                self.trigrams[tg] += 1
                self.next2_given[w[i : i + 2]][w[i + 2]] += 1
            # Soft syllable-like chunks from brands (CV+ patterns), frequency-weighted
            for chunk in re_find_chunks(w):
                if 2 <= len(chunk) <= 4:
                    self.syllable_mass[chunk] += 1
            # Approximate phoneme stream = vowels/consonants classes + letters
            prev = "^"
            for ch in w:
                p = "V" if is_vowel(ch) else "C"
                self.phoneme_trans[prev][p + ch] += 1
                prev = p + ch
            self.phoneme_trans[prev]["$"] += 1
            # Structural CV templates + positional letter mass (premium short brands)
            if 5 <= len(w) <= 8:
                tmpl = "".join("V" if is_vowel(c) else "C" for c in w)
                self.templates[tmpl] += 1
                for i, ch in enumerate(w):
                    self.pos_chars[(tmpl, i)][ch] += 1

    def _build_transformer_weights(self) -> None:
        """Tiny character embedding + self-attention prior over brand contexts."""
        alphabet = sorted(set("".join(self.brands)) | set("abcdefghijklmnopqrstuvwxyz"))
        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for c, i in self.stoi.items()}
        d = 16
        n = len(alphabet)
        rng = np.random.default_rng(stable_seed("|".join(self.brands[:20]), bits=32))
        self.E = rng.normal(0, 0.1, size=(n, d))
        # Fit embeddings toward co-occurrence: simple PMI-ish nudge
        for bg, cnt in self.bigrams.items():
            if bg[0] in self.stoi and bg[1] in self.stoi:
                a, b = self.stoi[bg[0]], self.stoi[bg[1]]
                self.E[a] += 0.01 * cnt * self.E[b]
                self.E[b] += 0.01 * cnt * self.E[a]
        # Normalize
        norms = np.linalg.norm(self.E, axis=1, keepdims=True) + 1e-9
        self.E = self.E / norms
        self.Wq = rng.normal(0, 0.05, size=(d, d))
        self.Wk = rng.normal(0, 0.05, size=(d, d))
        self.Wv = rng.normal(0, 0.05, size=(d, d))
        self.Wout = rng.normal(0, 0.05, size=(d, n))

    def transformer_next_probs(self, context: str) -> dict[str, float]:
        if not context:
            context = "a"
        ids = [self.stoi[c] for c in context[-6:] if c in self.stoi]
        if not ids:
            return {c: 1.0 for c in self.itos.values()}
        X = self.E[ids]  # t x d
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        scale = math.sqrt(Q.shape[1])
        attn = Q @ K.T / scale
        attn = attn - attn.max(axis=1, keepdims=True)
        weights = np.exp(attn)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)
        ctx = (weights @ V).mean(axis=0)
        logits = ctx @ self.Wout
        # Blend with empirical next-char distribution
        empir = self.next_given.get(context[-1], Counter())
        for ch, c in empir.items():
            if ch in self.stoi:
                logits[self.stoi[ch]] += math.log(1 + c)
        logits = logits - logits.max()
        ex = np.exp(logits)
        probs = ex / (ex.sum() + 1e-9)
        return {self.itos[i]: float(probs[i]) for i in range(len(self.itos))}

    def sample_length(self, lo: int, hi: int) -> int:
        # Bias toward punchy 5–7 letter premium brands
        options = [L for L in range(lo, hi + 1)]
        weights = []
        for L in options:
            base = self.length_hist[L] + 1
            if 5 <= L <= 7:
                base *= 4
            elif L == 8:
                base *= 2
            weights.append(base)
        return self.rng.choices(options, weights=weights, k=1)[0]

    def char_lm_logprob(self, name: str) -> float:
        """Character LM score with trigram interpolation."""
        if not name:
            return -1e9
        lp = math.log((self.starts[name[0]] + 0.5) / (sum(self.starts.values()) + 13))
        for i in range(len(name)):
            if i == 0:
                continue
            if i >= 2:
                ctx2 = name[i - 2 : i]
                c = self.next2_given[ctx2][name[i]]
                z = sum(self.next2_given[ctx2].values()) + 26
                p2 = (c + 0.1) / z
            else:
                p2 = 0.0
            ctx1 = name[i - 1]
            c1 = self.next_given[ctx1][name[i]]
            z1 = sum(self.next_given[ctx1].values()) + 26
            p1 = (c1 + 0.1) / z1
            p0 = (self.unigrams[name[i]] + 0.1) / (sum(self.unigrams.values()) + 26)
            p = 0.5 * p2 + 0.35 * p1 + 0.15 * p0 if p2 else 0.7 * p1 + 0.3 * p0
            lp += math.log(max(p, 1e-12))
        # Ending prior
        end = name[-2:] if len(name) >= 2 else name
        lp += 0.35 * math.log((self.ends[end] + 0.2) / (sum(self.ends.values()) + 50))
        return lp

    def weighted_chunks(self) -> list[tuple[str, float]]:
        total = sum(self.syllable_mass.values()) or 1
        return [(s, c / total) for s, c in self.syllable_mass.items() if c >= 2]


def re_find_chunks(word: str) -> list[str]:
    """Extract brand-like orthographic chunks (not a fixed dictionary)."""
    import re

    return re.findall(r"[^aeiouy]*[aeiouy]+[^aeiouy]*", word)

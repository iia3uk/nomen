"""Human simulation — 100 virtual users run pairwise brand preference tournaments."""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from nomen.models import Candidate


@dataclass
class TournamentResult:
    winners: list[Candidate]
    eliminated: list[Candidate]
    votes: dict[str, int]


def _stable_seed(*parts: str) -> int:
    return zlib.adler32("|".join(parts).encode("utf-8")) & 0xFFFFFFFF


def _user_utility(c: Candidate, rng: random.Random) -> float:
    """One virtual user's preference: beauty-first with noisy secondary tastes."""
    s = c.scores
    w_beauty = rng.uniform(0.32, 0.48)
    w_brand = rng.uniform(0.20, 0.34)
    w_premium = rng.uniform(0.08, 0.18)
    w_mem = rng.uniform(0.05, 0.14)
    w_novel = rng.uniform(0.02, 0.10)  # novelty is a spice, not the meal
    w_coll = rng.uniform(0.02, 0.10)
    raw = (
        w_beauty * s.beauty_score
        + w_brand * s.brand_score
        + w_premium * s.premium_feel
        + w_mem * s.memorability
        + w_novel * s.novelty_score
        + w_coll * s.collision_score
    )
    return raw + rng.gauss(0, 2.5)


def pairwise_prefer(a: Candidate, b: Candidate, n_users: int = 100, seed: int = 0) -> Candidate:
    """Which brand would you rather trust? Majority of n_users."""
    rng = random.Random(seed ^ _stable_seed(a.name, b.name))
    votes_a = 0
    for i in range(n_users):
        u = random.Random(rng.randint(0, 2**31 - 1) + i * 17)
        if _user_utility(a, u) >= _user_utility(b, u):
            votes_a += 1
    return a if votes_a >= (n_users - votes_a) else b


def run_tournament(
    candidates: list[Candidate],
    *,
    keep: int,
    n_users: int = 100,
    seed: int = 42,
) -> TournamentResult:
    """Pairwise preference tournament; keep top brands by win count + beauty."""
    if len(candidates) <= keep:
        return TournamentResult(winners=list(candidates), eliminated=[], votes={})

    rng = random.Random(seed)
    pool = list(candidates)
    wins: dict[str, int] = {c.name: 0 for c in pool}
    max_opponents = min(12, len(pool) - 1)
    seen_pairs: set[tuple[str, str]] = set()

    for c in pool:
        others = [x for x in pool if x.name != c.name]
        rng.shuffle(others)
        fought = 0
        for opp in others:
            pair = tuple(sorted((c.name, opp.name)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            match_seed = seed ^ _stable_seed(pair[0], pair[1]) ^ (fought * 997)
            winner = pairwise_prefer(c, opp, n_users=n_users, seed=match_seed)
            wins[winner.name] = wins.get(winner.name, 0) + 1
            fought += 1
            if fought >= max_opponents:
                break

    ranked = sorted(
        pool,
        key=lambda c: (
            wins.get(c.name, 0),
            c.scores.beauty_score,
            c.scores.overall,
            c.scores.brand_score,
        ),
        reverse=True,
    )
    return TournamentResult(
        winners=ranked[:keep],
        eliminated=ranked[keep:],
        votes=wins,
    )

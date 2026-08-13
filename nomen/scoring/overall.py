"""Multi-objective overall score. Beauty dominates; novelty must not."""

from __future__ import annotations

# BeautyScore 35% · BrandScore 30% · NoveltyScore 20% · CollisionScore 15%
W_BEAUTY = 0.35
W_BRAND = 0.30
W_NOVELTY = 0.20
W_COLLISION = 0.15


def compute_overall(
    *,
    beauty: float,
    brand: float,
    novelty: float,
    collision: float,
) -> float:
    overall = (
        W_BEAUTY * beauty
        + W_BRAND * brand
        + W_NOVELTY * novelty
        + W_COLLISION * collision
    )
    return round(max(0.0, min(100.0, overall)), 2)

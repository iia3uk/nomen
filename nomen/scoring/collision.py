"""CollisionScore — uniqueness / collision-risk objective (higher = safer).

Independent of BeautyScore. Uses distance to known brand orthography space
and length heuristics. Does not call live registries or search APIs.
"""

from __future__ import annotations

from functools import lru_cache

from rapidfuzz.distance import JaroWinkler, Levenshtein

from nomen.linguistics import normalize, occupied_brands


@lru_cache(maxsize=1)
def _known_brands() -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for w in occupied_brands():
        w = normalize(w)
        if 4 <= len(w) <= 12 and w not in seen:
            seen.add(w)
            out.append(w)
    return tuple(out)


def collision_score(name: str) -> float:
    """0–100. High means low collision risk with known software brands."""
    n = normalize(name)
    if not n:
        return 0.0

    # Short names collide more on the open web / app stores
    length_penalty = max(0.0, 8 - len(n)) * 6.5

    brands = _known_brands()
    # Sample by length neighborhood for speed
    near = [b for b in brands if abs(len(b) - len(n)) <= 2]
    if not near:
        near = list(brands[:200])

    best_sim = 0.0
    for b in near:
        if n == b:
            return 0.0
        lev = 1.0 - Levenshtein.normalized_distance(n, b)
        jw = JaroWinkler.similarity(n, b)
        sim = 0.55 * lev + 0.45 * jw
        if sim > best_sim:
            best_sim = sim
        if best_sim >= 0.92:
            break

    # Convert similarity-to-known into safety score
    proximity_penalty = best_sim * 70.0
    # Mild boost for uncommon length in brand space
    score = 100.0 - length_penalty - proximity_penalty
    if best_sim < 0.55:
        score += 8
    return round(max(0.0, min(100.0, score)), 2)

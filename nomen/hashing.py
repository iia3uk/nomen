"""Process-stable hashes — Python's builtin hash() is randomized per interpreter."""

from __future__ import annotations

import hashlib


def stable_seed(text: str, *, bits: int = 31) -> int:
    """Deterministic integer seed from a string (blake2b, independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (1 << bits)

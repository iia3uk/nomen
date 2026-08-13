"""Permanent elite archive of the best brands ever generated.

A new candidate replaces an archived name only if it is objectively better
(higher overall, and not worse on beauty).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import orjson

from nomen.linguistics import normalize, reserved_brands
from nomen.models import Candidate
from nomen.persist import atomic_write_bytes


class EliteArchive:
    def __init__(self, path: Path, *, capacity: int = 64) -> None:
        self.path = path
        self.capacity = capacity
        self.entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = orjson.loads(self.path.read_bytes())
                if isinstance(data, list):
                    self.entries = data
            except Exception as exc:
                print(f"elite archive unreadable ({exc}); starting empty", file=sys.stderr)
                self.entries = []
        reserved = reserved_brands()
        if reserved:
            kept = [
                e
                for e in self.entries
                if normalize(str(e.get("name", ""))) not in reserved
            ]
            if len(kept) != len(self.entries):
                self.entries = kept
                self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(self.path, orjson.dumps(self.entries, option=orjson.OPT_INDENT_2))

    def names(self) -> list[str]:
        return [e["name"] for e in self.entries if "name" in e]

    def _row(self, c: Candidate) -> dict[str, Any]:
        return {
            "name": c.name,
            "beauty_score": c.scores.beauty_score,
            "brand_score": c.scores.brand_score,
            "novelty_score": c.scores.novelty_score,
            "collision_score": c.scores.collision_score,
            "overall": c.scores.overall,
            "generator": c.generator,
            "phonetic_root": c.scores.phonetic_root,
        }

    @staticmethod
    def _better(new: dict[str, Any], old: dict[str, Any]) -> bool:
        """Objectively better: higher overall, beauty not worse (within 1pt slack)."""
        if new["overall"] <= old["overall"]:
            return False
        if new["beauty_score"] + 1.0 < old["beauty_score"]:
            return False
        # Prefer clear beauty wins when overall is close
        if new["overall"] < old["overall"] + 1.5 and new["beauty_score"] <= old["beauty_score"]:
            return False
        return True

    def consider(self, candidate: Candidate) -> bool:
        """Try to insert candidate. Returns True if archive changed."""
        if normalize(candidate.name) in reserved_brands():
            return False
        row = self._row(candidate)
        # Exact name already present — upgrade in place if better
        for i, e in enumerate(self.entries):
            if e.get("name") == row["name"]:
                if self._better(row, e):
                    self.entries[i] = row
                    self._sort_trim()
                    return True
                return False

        if len(self.entries) < self.capacity:
            self.entries.append(row)
            self._sort_trim()
            return True

        # Must beat the weakest elite member
        weakest_i = min(
            range(len(self.entries)),
            key=lambda i: (
                self.entries[i].get("overall", 0),
                self.entries[i].get("beauty_score", 0),
            ),
        )
        weakest = self.entries[weakest_i]
        if self._better(row, weakest):
            self.entries[weakest_i] = row
            self._sort_trim()
            return True
        return False

    def consider_many(self, candidates: list[Candidate]) -> int:
        n = 0
        for c in candidates:
            if self.consider(c):
                n += 1
        if n:
            self.save()
        return n

    def _sort_trim(self) -> None:
        self.entries.sort(
            key=lambda e: (e.get("overall", 0), e.get("beauty_score", 0)),
            reverse=True,
        )
        self.entries = self.entries[: self.capacity]

    def must_beat_floor(self) -> float:
        """Minimum overall a new shortlist member should aim for vs elite."""
        if not self.entries:
            return 0.0
        if len(self.entries) < self.capacity:
            return 0.0
        return min(e.get("overall", 0) for e in self.entries)

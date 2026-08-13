"""Checkpoint persistence."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import orjson

from nomen.models import HuntState
from nomen.persist import atomic_write_bytes


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self, seed: str) -> HuntState | None:
        if not self.path.exists():
            return None
        try:
            state = HuntState.model_validate(orjson.loads(self.path.read_bytes()))
            return state if state.seed == seed else None
        except Exception as exc:
            print(f"checkpoint unreadable ({exc}); starting fresh", file=sys.stderr)
            return None

    def save(self, state: HuntState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        atomic_write_bytes(
            self.path,
            orjson.dumps(state.model_dump(mode="json"), option=orjson.OPT_INDENT_2),
        )

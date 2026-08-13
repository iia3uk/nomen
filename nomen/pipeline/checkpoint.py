"""Checkpoint persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import orjson

from nomen.models import HuntState


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
        except Exception:
            return None

    def save(self, state: HuntState) -> None:
        state.updated_at = datetime.now(timezone.utc)
        self.path.write_bytes(orjson.dumps(state.model_dump(mode="json"), option=orjson.OPT_INDENT_2))

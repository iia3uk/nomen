"""Domain models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class Stage(str, Enum):
    QUALITY = "quality"
    BANNED = "banned"
    ENGLISH = "english"
    SIMILARITY = "similarity"
    DIVERSITY = "diversity"
    NOVELTY = "novelty"
    BEAUTY = "beauty"
    SCORE = "score"
    TOURNAMENT = "tournament"
    REGISTRY = "registry"
    GITHUB = "github"
    DOMAIN = "domain"
    COMPANY = "company"
    SEARCH = "search"
    TRADEMARK = "trademark"


class Scores(BaseModel):
    pronounceability: float = 0.0
    memorability: float = 0.0
    typing_speed: float = 0.0
    visual_balance: float = 0.0
    brand_strength: float = 0.0
    premium_feel: float = 0.0
    international_readability: float = 0.0
    collision_probability: float = 0.0
    seo_uniqueness: float = 0.0
    beauty_score: float = 0.0
    brand_score: float = 0.0
    novelty_score: float = 0.0
    collision_score: float = 0.0
    overall: float = 0.0
    cv_pattern: str = ""
    phonetic_root: str = ""


class Candidate(BaseModel):
    name: str
    generator: str
    scores: Scores = Field(default_factory=Scores)
    rejection_reasons: list[str] = Field(default_factory=list)
    rejected_at: Stage | None = None
    clean: bool = False
    registries: dict[str, Any] = Field(default_factory=dict)
    domains_registered: list[str] = Field(default_factory=list)
    search_hits: dict[str, Any] = Field(default_factory=dict)
    company_hits: int | None = None
    trademark_hits: int | None = None
    errors: list[str] = Field(default_factory=list)
    checked_at: datetime | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def display_name(self) -> str:
        return self.name[:1].upper() + self.name[1:]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def variants(self) -> list[str]:
        d = self.display_name
        return [f"{d} CMS", f"{d} Platform", f"{d} OS", f"{d} Engine"]

    def reject(self, stage: Stage, reason: str) -> Candidate:
        self.clean = False
        self.rejected_at = stage
        self.rejection_reasons.append(f"{stage.value}: {reason}")
        return self

    def mark_clean(self) -> Candidate:
        self.clean = True
        self.rejected_at = None
        self.checked_at = datetime.now(timezone.utc)
        return self


class HuntState(BaseModel):
    seed: str
    round: int = 0
    generated_total: int = 0
    checked_total: int = 0
    clean_names: list[str] = Field(default_factory=list)
    winner_pool: list[str] = Field(default_factory=list)
    archive_names: list[str] = Field(default_factory=list)
    exploration_restarts: int = 0
    rejected: dict[str, list[str]] = Field(default_factory=dict)
    pending: list[str] = Field(default_factory=list)
    candidate_meta: dict[str, dict[str, Any]] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

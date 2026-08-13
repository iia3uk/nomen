"""Configuration loading (YAML + environment)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
DEFAULT_CONFIG = Path("config.yaml")


class SimilarityConfig(BaseModel):
    levenshtein_max: int = 2
    damerau_max: int = 2
    jaro_winkler_min: float = 0.88
    ngram_min: float = 0.72
    embedding_cosine_min: float = 0.82
    metaphone_match: bool = True


class EmbeddingsConfig(BaseModel):
    enabled: bool = True
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class GenerationConfig(BaseModel):
    rounds_max: int = 0
    evolution_generations: int = 8
    winner_pool: int = 64
    workers: int = 0  # 0 = auto (cpu_count - 1)


class AppConfig(BaseModel):
    seed: str = "ai-native-cms-mcp-platform"
    generate_batch: int = 50_000
    online_batch: int = 40
    target_clean: int = 20
    min_overall_score: float = 92.0
    min_beauty_score: float = 72.0
    min_len: int = 5
    max_len: int = 9
    strict: bool = True
    resume: bool = True
    concurrency: int = 32
    pause_seconds: float = 0.05
    timeout: float = 15.0
    workers: int = 0  # process pool size for generators; 0 = auto
    output_dir: str = "nomen_results"
    tlds: list[str] = Field(
        default_factory=lambda: ["com", "io", "dev", "ai", "app", "tech", "cloud", "so"]
    )
    similarity: SimilarityConfig = Field(default_factory=SimilarityConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class Secrets(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")
    brave_search_api_key: str | None = Field(default=None, validation_alias="BRAVE_SEARCH_API_KEY")
    serpapi_key: str | None = Field(default=None, validation_alias="SERPAPI_KEY")
    bing_search_api_key: str | None = Field(default=None, validation_alias="BING_SEARCH_API_KEY")
    opencorporates_api_token: str | None = Field(
        default=None, validation_alias="OPENCORPORATES_API_TOKEN"
    )
    whois_api_key: str | None = Field(default=None, validation_alias="WHOIS_API_KEY")

    @property
    def has_web_search(self) -> bool:
        return bool(self.brave_search_api_key or self.serpapi_key or self.bing_search_api_key)


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    cfg_path = path or DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"similarity", "embeddings", "generation"} and isinstance(value, dict):
                merged = {**(raw.get(key) or {}), **value}
                raw[key] = merged
            else:
                raw[key] = value
    return AppConfig.model_validate(raw)

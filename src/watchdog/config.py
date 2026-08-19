"""Loads config.yaml and .env into a single typed object used across the pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    shipments: Path
    notes: Path
    sample_format: Path
    output: Path


@dataclass(frozen=True)
class BaselineConfig:
    own_history_weeks: int
    peer_scope: str
    peer_excludes_self: bool


@dataclass(frozen=True)
class DetectorConfig:
    own_history_threshold_pct: float
    peer_threshold_pct: float


@dataclass(frozen=True)
class NotesConfig:
    open_ended_note_weeks: int
    near_miss_days: int


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    max_retries: int
    retry_backoff_seconds: float
    api_key: str | None


@dataclass(frozen=True)
class RetrievalConfig:
    backend: str
    top_k: int
    embedding_model: str


@dataclass(frozen=True)
class Config:
    paths: Paths
    baselines: BaselineConfig
    detector: DetectorConfig
    notes: NotesConfig
    llm: LLMConfig
    retrieval: RetrievalConfig


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> Config:
    load_dotenv(REPO_ROOT / ".env")
    with open(path) as f:
        raw = yaml.safe_load(f)

    root = Path(path).resolve().parent
    paths = Paths(**{k: root / v for k, v in raw["paths"].items()})
    baselines = BaselineConfig(**raw["baselines"])
    detector = DetectorConfig(**raw["detector"])
    notes = NotesConfig(**raw["notes"])
    llm_raw = raw["llm"]
    llm = LLMConfig(**llm_raw, api_key=os.environ.get("NVIDIA_API_KEY"))
    retrieval = RetrievalConfig(**raw["retrieval"])

    return Config(paths=paths, baselines=baselines, detector=detector, notes=notes, llm=llm, retrieval=retrieval)

"""Feature engineering for commit prediction."""

import math
from typing import List
from datetime import datetime, timezone

REPO_VOCAB: List[str] = [
    "plato-training", "plato-types", "tensor-spline", "plato-data",
    "constraint-theory-core", "constraint-theory-py", "cocapn-ai-web",
    "forgemaster", "plato-vessel-core", "casting-call",
    "constraint-inference", "intent-inference", "holonomy-consensus",
    "flux-lucid", "dodecet-encoder", "neural-plato",
    "openarm", "plato-model-ocean", "plato-escalation-gate",
    "plato-room-intelligence", "spectral-conservation",
]

LANG_VOCAB: List[str] = [
    ".py", ".rs", ".js", ".ts", ".md", ".toml", ".json",
    ".c", ".cpp", ".h",
]


def hour_features(timestamp: float) -> List[float]:
    """Return [sin, cos] cyclical encoding of hour-of-day."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    h: int = dt.hour
    return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]


def day_features(timestamp: float) -> List[float]:
    """Return [sin, cos] cyclical encoding of day-of-week."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    d: int = dt.weekday()
    return [math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7)]


def repo_onehot(repo: str) -> List[float]:
    """One-hot encode a repo name against REPO_VOCAB."""
    vec: List[float] = [0.0] * len(REPO_VOCAB)
    if repo in REPO_VOCAB:
        vec[REPO_VOCAB.index(repo)] = 1.0
    return vec


def lang_features(languages: List[str]) -> List[float]:
    """Multi-hot encode file extensions against LANG_VOCAB."""
    vec: List[float] = [0.0] * len(LANG_VOCAB)
    for lang in languages:
        ext: str = lang if lang.startswith(".") else f".{lang}"
        if ext in LANG_VOCAB:
            vec[LANG_VOCAB.index(ext)] = 1.0
    return vec

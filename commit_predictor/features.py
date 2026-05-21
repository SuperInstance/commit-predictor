"""Feature engineering for commit prediction."""

import math
import numpy as np
from typing import List
from datetime import datetime, timezone

REPO_VOCAB = [
    "plato-training", "plato-types", "tensor-spline", "plato-data",
    "constraint-theory-core", "constraint-theory-py", "cocapn-ai-web",
    "forgemaster", "plato-vessel-core", "casting-call",
    "constraint-inference", "intent-inference", "holonomy-consensus",
    "flux-lucid", "dodecet-encoder", "neural-plato",
    "openarm", "plato-model-ocean", "plato-escalation-gate",
    "plato-room-intelligence", "spectral-conservation",
]

LANG_VOCAB = [".py", ".rs", ".js", ".ts", ".md", ".toml", ".json", ".c", ".cpp", ".h"]


def hour_features(timestamp: float) -> List[float]:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    h = dt.hour
    return [math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)]


def day_features(timestamp: float) -> List[float]:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    d = dt.weekday()
    return [math.sin(2 * math.pi * d / 7), math.cos(2 * math.pi * d / 7)]


def repo_onehot(repo: str) -> List[float]:
    vec = [0.0] * len(REPO_VOCAB)
    if repo in REPO_VOCAB:
        vec[REPO_VOCAB.index(repo)] = 1.0
    return vec


def lang_features(languages: List[str]) -> List[float]:
    vec = [0.0] * len(LANG_VOCAB)
    for lang in languages:
        ext = lang if lang.startswith(".") else f".{lang}"
        if ext in LANG_VOCAB:
            vec[LANG_VOCAB.index(ext)] = 1.0
    return vec

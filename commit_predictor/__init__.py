"""Commit Predictor — forecast fleet git activity with a tiny dense network."""

from .features import (
    hour_features, day_features, repo_onehot, lang_features,
    REPO_VOCAB, LANG_VOCAB,
)
from .data import CommitPoint, PredictionSample, build_prediction_dataset
from .model import CommitPredictor
from .pipeline import train_commit_predictor

__all__ = [
    "hour_features", "day_features", "repo_onehot", "lang_features",
    "REPO_VOCAB", "LANG_VOCAB",
    "CommitPoint", "PredictionSample", "build_prediction_dataset",
    "CommitPredictor",
    "train_commit_predictor",
]

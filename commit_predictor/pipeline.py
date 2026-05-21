"""End-to-end training pipeline."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .data import CommitPoint, build_prediction_dataset
from .model import CommitPredictor


def train_commit_predictor(
    commits: List[CommitPoint],
    repos: Optional[List[str]] = None,
    window_hours: float = 6.0,
    epochs: int = 100,
    hidden_dim: int = 32,
    lr: float = 0.01,
) -> Tuple[CommitPredictor, Dict]:
    target_repos = repos or list(set(c.repo for c in commits))
    all_samples = []
    for repo in target_repos:
        all_samples.extend(build_prediction_dataset(commits, repo, window_hours=window_hours))
    if not all_samples:
        raise ValueError("No training samples generated")

    feature_dim = all_samples[0].features.shape[0]
    X = np.stack([s.features for s in all_samples])
    y_commit = np.array([s.label_commit_next_hour for s in all_samples], dtype=np.float32)
    y_files = np.array([s.label_file_count for s in all_samples], dtype=np.float32)
    y_crossref = np.array([s.label_cross_ref for s in all_samples], dtype=np.float32)

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X = (X - mean) / std
    y_files = (y_files > 0).astype(np.float32)

    model = CommitPredictor(input_dim=feature_dim, hidden_dim=hidden_dim, lr=lr)
    model.fit(X, y_commit, y_files, y_crossref, epochs=epochs, verbose=True)

    preds = model.predict(X)
    metrics = {
        "samples": len(all_samples), "repos": len(target_repos), "feature_dim": feature_dim,
        "commit_accuracy": round(float(np.mean((preds["commit_prob"] > 0.5) == (y_commit > 0.5))), 4),
        "file_activity_accuracy": round(float(np.mean((preds["file_count"] > 0.5) == (y_files > 0.5))), 4),
        "crossref_accuracy": round(float(np.mean((preds["crossref_prob"] > 0.5) == (y_crossref > 0.5))), 4),
        "final_loss": round(model.losses[-1], 4),
    }
    return model, metrics

"""Data structures and dataset builder."""

import math
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .features import REPO_VOCAB, LANG_VOCAB, hour_features, day_features


@dataclass
class CommitPoint:
    """A single commit as a multidimensional data point."""
    sha: str
    repo: str
    author: str
    timestamp: float
    message: str
    files_changed: int
    insertions: int
    deletions: int
    is_merge: bool
    cross_refs: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.insertions + self.deletions

    @property
    def net_lines(self) -> int:
        return self.insertions - self.deletions


@dataclass
class PredictionSample:
    features: np.ndarray
    label_commit_next_hour: float
    label_file_count: float
    label_cross_ref: float
    repo: str
    window_start: float
    window_end: float


def build_prediction_dataset(
    commits: List[CommitPoint],
    repo: str,
    window_hours: float = 6.0,
    step_hours: float = 1.0,
    lookahead_hours: float = 1.0,
) -> List[PredictionSample]:
    if not commits:
        return []

    repo_commits = sorted(
        [c for c in commits if c.repo == repo],
        key=lambda c: c.timestamp,
    )
    if not repo_commits:
        return []

    min_ts = repo_commits[0].timestamp
    max_ts = repo_commits[-1].timestamp
    window_s = window_hours * 3600
    step_s = step_hours * 3600
    lookahead_s = lookahead_hours * 3600

    samples = []
    t = min_ts + window_s

    while t + lookahead_s <= max_ts:
        window_commits = [c for c in repo_commits if t - window_s <= c.timestamp < t]
        lookahead_commits = [c for c in repo_commits if t <= c.timestamp < t + lookahead_s]

        agg = np.zeros(7, dtype=np.float32)
        if window_commits:
            agg[0] = len(window_commits)
            agg[1] = sum(c.files_changed for c in window_commits)
            agg[2] = sum(c.insertions for c in window_commits)
            agg[3] = sum(c.deletions for c in window_commits)
            agg[4] = sum(len(c.cross_refs) for c in window_commits)
            hours = [datetime.fromtimestamp(c.timestamp, tz=timezone.utc).hour for c in window_commits]
            agg[5] = math.sin(2 * math.pi * np.mean(hours) / 24)
            agg[6] = math.cos(2 * math.pi * np.mean(hours) / 24)

        tf = np.array(hour_features(t) + day_features(t), dtype=np.float32)
        features = np.concatenate([tf, agg])

        has_commit = 1.0 if lookahead_commits else 0.0
        file_count = float(sum(c.files_changed for c in lookahead_commits))
        has_crossref = 1.0 if any(c.cross_refs for c in lookahead_commits) else 0.0

        samples.append(PredictionSample(
            features=features, label_commit_next_hour=has_commit,
            label_file_count=file_count, label_cross_ref=has_crossref,
            repo=repo, window_start=t - window_s, window_end=t,
        ))
        t += step_s

    return samples

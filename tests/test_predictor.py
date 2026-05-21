"""Tests for commit-predictor."""

import time
import tempfile
import numpy as np
import pytest
from commit_predictor import (
    hour_features, day_features, repo_onehot, lang_features,
    CommitPoint, build_prediction_dataset, CommitPredictor,
)


class TestFeatures:
    def test_hour_features_range(self):
        for h in range(24):
            ts = 1700000000.0 + h * 3600
            sf, cf = hour_features(ts)
            assert -1.0 <= sf <= 1.0
            assert -1.0 <= cf <= 1.0
            assert abs(sf**2 + cf**2 - 1.0) < 1e-6

    def test_day_features_cycle(self):
        # Monday vs Friday should differ
        from datetime import datetime, timezone
        mon = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        fri = datetime(2024, 1, 5, tzinfo=timezone.utc).timestamp()
        assert day_features(mon) != day_features(fri)

    def test_repo_onehot(self):
        v = repo_onehot("plato-training")
        assert sum(v) == 1.0
        assert v[0] == 1.0

    def test_repo_unknown(self):
        v = repo_onehot("nonexistent-repo")
        assert sum(v) == 0.0

    def test_lang_features(self):
        v = lang_features([".py", ".rs"])
        assert sum(v) == 2.0


class TestCommitPoint:
    def test_size(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=1, insertions=10, deletions=5, is_merge=False)
        assert c.size == 15

    def test_net_lines(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=1, insertions=10, deletions=3, is_merge=False)
        assert c.net_lines == 7


class TestDataset:
    def test_build_empty(self):
        assert build_prediction_dataset([], "repo") == []

    def test_build_with_commits(self):
        base = 1700000000.0
        commits = [
            CommitPoint(sha=f"c{i}", repo="r", author="a", timestamp=base + i * 1800,
                        message="m", files_changed=1, insertions=5, deletions=2,
                        is_merge=False, languages=[".py"])
            for i in range(30)
        ]
        samples = build_prediction_dataset(commits, "r", window_hours=3.0, step_hours=1.0)
        assert len(samples) > 0
        assert samples[0].features.shape[0] > 0


class TestCommitPredictor:
    def test_forward_shape(self):
        model = CommitPredictor(input_dim=11, hidden_dim=16)
        X = np.random.randn(4, 11).astype(np.float32)
        c, f, x = model.forward(X)
        assert c.shape == (4, 1)
        assert f.shape == (4, 1)
        assert x.shape == (4, 1)

    def test_save_load_roundtrip(self):
        model = CommitPredictor(input_dim=11, hidden_dim=8)
        X = np.random.randn(2, 11).astype(np.float32)
        pred_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            model.save(f.name)
            loaded = CommitPredictor.load(f.name)

        pred_after = loaded.predict(X)
        np.testing.assert_allclose(pred_before["commit_prob"], pred_after["commit_prob"], atol=1e-5)

    def test_train_runs(self):
        model = CommitPredictor(input_dim=11, hidden_dim=16, lr=0.01)
        np.random.seed(42)
        X = np.random.randn(50, 11).astype(np.float32)
        y = (np.random.rand(50) > 0.5).astype(np.float32)
        losses = model.fit(X, y, y, y, epochs=50, batch_size=25, verbose=False)
        assert len(losses) == 50
        assert all(np.isfinite(l) for l in losses)

"""Tests for commit_predictor — features, model, data, pipeline."""

import time
import pytest
import numpy as np
from commit_predictor.features import (
    hour_features, day_features, repo_onehot, LANG_VOCAB, REPO_VOCAB,
)
from commit_predictor.model import CommitPredictor
from commit_predictor.data import CommitPoint


class TestHourFeatures:
    def test_output_shape(self):
        feats = hour_features(time.time())
        assert len(feats) == 2

    def test_range(self):
        feats = hour_features(time.time())
        assert -1.0 <= feats[0] <= 1.0
        assert -1.0 <= feats[1] <= 1.0

    def test_different_hours(self):
        # Noon vs midnight should give different values
        noon = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        midnight = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        f_noon = hour_features(noon)
        f_midnight = hour_features(midnight)
        assert f_noon != f_midnight


class TestDayFeatures:
    def test_output_shape(self):
        feats = day_features(time.time())
        assert len(feats) == 2

    def test_range(self):
        feats = day_features(time.time())
        assert -1.0 <= feats[0] <= 1.0


class TestRepoOnehot:
    def test_known_repo(self):
        vec = repo_onehot("plato-training")
        assert len(vec) == len(REPO_VOCAB)
        assert 1.0 in vec

    def test_unknown_repo(self):
        vec = repo_onehot("nonexistent-repo")
        assert all(v == 0.0 for v in vec)


class TestCommitPredictor:
    def test_creation(self):
        model = CommitPredictor(input_dim=10, hidden_dim=16, lr=0.01)
        assert model.input_dim == 10
        assert model.hidden_dim == 16

    def test_forward_shape(self):
        model = CommitPredictor(input_dim=10, hidden_dim=16)
        X = np.random.randn(5, 10).astype(np.float32)
        commit_out, files_out, crossref_out = model.forward(X)
        assert commit_out.shape == (5, 1)
        assert files_out.shape == (5, 1)
        assert crossref_out.shape == (5, 1)

    def test_predict_keys(self):
        model = CommitPredictor(input_dim=10, hidden_dim=16)
        X = np.random.randn(3, 10).astype(np.float32)
        preds = model.predict(X)
        assert "commit_prob" in preds
        assert "file_count" in preds
        assert "crossref_prob" in preds

    def test_fit_reduces_loss(self):
        model = CommitPredictor(input_dim=5, hidden_dim=8, lr=0.1)
        X = np.random.randn(20, 5).astype(np.float32)
        y_c = np.random.randint(0, 2, 20).astype(np.float32)
        y_f = np.random.rand(20).astype(np.float32)
        y_x = np.random.randint(0, 2, 20).astype(np.float32)
        model.fit(X, y_c, y_f, y_x, epochs=10, verbose=False)
        assert len(model.losses) > 0

    def test_repr(self):
        model = CommitPredictor(input_dim=10)
        r = repr(model)
        assert "CommitPredictor" in r


class TestCommitPoint:
    def test_size(self):
        cp = CommitPoint(
            sha="abc123", repo="test", author="dev",
            timestamp=time.time(), message="test",
            files_changed=5, insertions=100, deletions=20,
            is_merge=False,
        )
        assert cp.size == 120

    def test_net_lines(self):
        cp = CommitPoint(
            sha="abc", repo="test", author="dev",
            timestamp=time.time(), message="test",
            files_changed=1, insertions=50, deletions=30,
            is_merge=False,
        )
        assert cp.net_lines == 20


from datetime import datetime, timezone

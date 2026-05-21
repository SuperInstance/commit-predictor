"""Extended tests for commit_predictor — pipeline, data, features, model coverage."""

import time
import tempfile
import numpy as np
import pytest
from datetime import datetime, timezone

from commit_predictor.features import (
    hour_features, day_features, repo_onehot, lang_features,
    REPO_VOCAB, LANG_VOCAB,
)
from commit_predictor.data import CommitPoint, PredictionSample, build_prediction_dataset
from commit_predictor.model import CommitPredictor
from commit_predictor.pipeline import train_commit_predictor


# --- Features ---

class TestHourFeaturesExtended:
    def test_midnight_known_values(self):
        ts = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp()
        sf, cf = hour_features(ts)
        assert abs(sf) < 1e-6  # sin(0) ≈ 0
        assert abs(cf - 1.0) < 1e-6  # cos(0) ≈ 1

    def test_noon_known_values(self):
        ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        sf, cf = hour_features(ts)
        assert abs(sf) < 1e-6  # sin(π) ≈ 0
        assert abs(cf - (-1.0)) < 1e-6  # cos(π) ≈ -1

    def test_unit_circle(self):
        for h in range(24):
            ts = datetime(2024, 1, 1, h, 0, 0, tzinfo=timezone.utc).timestamp()
            sf, cf = hour_features(ts)
            assert abs(sf**2 + cf**2 - 1.0) < 1e-10


class TestDayFeaturesExtended:
    def test_monday_vs_sunday(self):
        mon = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()  # Monday
        sun = datetime(2024, 1, 7, tzinfo=timezone.utc).timestamp()  # Sunday
        assert day_features(mon) != day_features(sun)

    def test_unit_circle(self):
        for d in range(7):
            ts = datetime(2024, 1, 1 + d, tzinfo=timezone.utc).timestamp()
            sf, cf = day_features(ts)
            assert abs(sf**2 + cf**2 - 1.0) < 1e-10


class TestRepoOnehotExtended:
    def test_all_repos_in_vocab(self):
        for repo in REPO_VOCAB:
            vec = repo_onehot(repo)
            assert sum(vec) == 1.0
            assert vec[REPO_VOCAB.index(repo)] == 1.0

    def test_length_matches_vocab(self):
        assert len(repo_onehot("any")) == len(REPO_VOCAB)


class TestLangFeaturesExtended:
    def test_dotted_and_plain(self):
        v1 = lang_features([".py"])
        v2 = lang_features(["py"])
        assert v1 == v2  # should normalize

    def test_unknown_lang_ignored(self):
        v = lang_features([".xyz"])
        assert sum(v) == 0.0

    def test_empty_list(self):
        v = lang_features([])
        assert sum(v) == 0.0
        assert len(v) == len(LANG_VOCAB)

    def test_all_langs(self):
        v = lang_features(LANG_VOCAB)
        assert sum(v) == len(LANG_VOCAB)


# --- Data ---

class TestCommitPointExtended:
    def test_cross_refs_default(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=1, insertions=10, deletions=5, is_merge=False)
        assert c.cross_refs == []

    def test_languages_default(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=1, insertions=10, deletions=5, is_merge=False)
        assert c.languages == []

    def test_size_zero(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=0, insertions=0, deletions=0, is_merge=False)
        assert c.size == 0

    def test_net_lines_negative(self):
        c = CommitPoint(sha="a", repo="r", author="x", timestamp=0, message="m",
                        files_changed=1, insertions=5, deletions=20, is_merge=False)
        assert c.net_lines == -15


class TestBuildPredictionDatasetExtended:
    def test_no_matching_repo(self):
        base = 1700000000.0
        commits = [
            CommitPoint(sha="c1", repo="repo-a", author="x", timestamp=base,
                        message="m", files_changed=1, insertions=5, deletions=2, is_merge=False)
        ]
        samples = build_prediction_dataset(commits, "repo-b")
        assert samples == []

    def test_single_commit(self):
        base = 1700000000.0
        commits = [
            CommitPoint(sha="c1", repo="r", author="x", timestamp=base,
                        message="m", files_changed=1, insertions=5, deletions=2, is_merge=False)
        ]
        samples = build_prediction_dataset(commits, "r", window_hours=0.1, step_hours=0.05)
        # With a single commit, window may not have enough span for lookahead
        assert isinstance(samples, list)

    def test_features_shape(self):
        base = 1700000000.0
        commits = [
            CommitPoint(sha=f"c{i}", repo="r", author="x", timestamp=base + i * 600,
                        message="m", files_changed=1, insertions=5, deletions=2,
                        is_merge=False, languages=[".py"])
            for i in range(50)
        ]
        samples = build_prediction_dataset(commits, "r", window_hours=2.0, step_hours=0.5)
        if samples:
            # Features: 4 (time) + 7 (aggregate) = 11
            assert samples[0].features.shape[0] == 11

    def test_labels_are_float(self):
        base = 1700000000.0
        commits = [
            CommitPoint(sha=f"c{i}", repo="r", author="x", timestamp=base + i * 600,
                        message="m", files_changed=1, insertions=5, deletions=2,
                        is_merge=False, cross_refs=["ref1"] if i % 5 == 0 else [])
            for i in range(50)
        ]
        samples = build_prediction_dataset(commits, "r", window_hours=2.0, step_hours=0.5)
        for s in samples:
            assert isinstance(s.label_commit_next_hour, float)
            assert isinstance(s.label_file_count, float)
            assert isinstance(s.label_cross_ref, float)


class TestPredictionSample:
    def test_fields(self):
        s = PredictionSample(
            features=np.zeros(11), label_commit_next_hour=1.0,
            label_file_count=3.0, label_cross_ref=0.0,
            repo="r", window_start=0.0, window_end=3600.0,
        )
        assert s.repo == "r"
        assert s.features.shape == (11,)


# --- Model ---

class TestCommitPredictorExtended:
    def test_sigmoid(self):
        assert abs(CommitPredictor.sigmoid(0) - 0.5) < 1e-6
        assert CommitPredictor.sigmoid(100) > 0.99
        assert CommitPredictor.sigmoid(-100) < 0.01

    def test_relu(self):
        assert CommitPredictor.relu(-1) == 0
        assert CommitPredictor.relu(0) == 0
        assert CommitPredictor.relu(5) == 5

    def test_predict_output_ranges(self):
        model = CommitPredictor(input_dim=10, hidden_dim=16)
        X = np.random.randn(10, 10).astype(np.float32)
        preds = model.predict(X)
        for key in preds:
            assert preds[key].min() >= 0.0
            assert preds[key].max() <= 1.0

    def test_train_step_returns_loss(self):
        model = CommitPredictor(input_dim=5, hidden_dim=8)
        X = np.random.randn(4, 5).astype(np.float32)
        y = np.array([1, 0, 1, 0], dtype=np.float32)
        loss = model.train_step(X, y, y, y)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_losses_tracked(self):
        model = CommitPredictor(input_dim=5, hidden_dim=8, lr=0.1)
        X = np.random.randn(20, 5).astype(np.float32)
        y = np.random.randint(0, 2, 20).astype(np.float32)
        model.fit(X, y, y, y, epochs=5, verbose=False)
        assert len(model.losses) == 5

    def test_save_load(self):
        model = CommitPredictor(input_dim=8, hidden_dim=4, lr=0.05)
        X = np.random.randn(3, 8).astype(np.float32)
        pred_before = model.predict(X)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            model.save(f.name)
            loaded = CommitPredictor.load(f.name)

        assert loaded.input_dim == 8
        assert loaded.hidden_dim == 4
        pred_after = loaded.predict(X)
        np.testing.assert_allclose(pred_before["commit_prob"], pred_after["commit_prob"], atol=1e-5)


# --- Pipeline ---

class TestPipeline:
    def _make_commits(self, n=50, repo="test-repo", base_ts=1700000000.0):
        return [
            CommitPoint(
                sha=f"c{i}", repo=repo, author="dev",
                timestamp=base_ts + i * 1200,
                message=f"commit {i}", files_changed=np.random.randint(1, 10),
                insertions=np.random.randint(5, 50),
                deletions=np.random.randint(1, 20),
                is_merge=False,
                cross_refs=[f"ref-{i}"] if i % 3 == 0 else [],
                languages=[".py"],
            )
            for i in range(n)
        ]

    def test_basic_training(self):
        commits = self._make_commits(60)
        model, metrics = train_commit_predictor(
            commits, repos=["test-repo"],
            window_hours=3.0, epochs=10, hidden_dim=8, lr=0.1,
        )
        assert isinstance(model, CommitPredictor)
        assert metrics["samples"] > 0
        assert metrics["repos"] == 1
        assert "commit_accuracy" in metrics
        assert "final_loss" in metrics

    def test_no_samples_raises(self):
        commits = self._make_commits(5)
        with pytest.raises(ValueError, match="No training samples"):
            train_commit_predictor(commits, repos=["test-repo"], window_hours=1.0, epochs=5)

    def test_multiple_repos(self):
        commits = (
            self._make_commits(40, repo="repo-a") +
            self._make_commits(40, repo="repo-b", base_ts=1700000000.0)
        )
        model, metrics = train_commit_predictor(
            commits, repos=["repo-a", "repo-b"],
            window_hours=3.0, epochs=5, hidden_dim=8,
        )
        assert metrics["repos"] == 2

    def test_auto_detect_repos(self):
        commits = self._make_commits(40, repo="auto-repo")
        model, metrics = train_commit_predictor(
            commits, window_hours=3.0, epochs=5, hidden_dim=8,
        )
        assert metrics["repos"] >= 1

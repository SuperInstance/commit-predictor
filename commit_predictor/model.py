"""Pure-numpy dense network for commit prediction."""

import numpy as np
from typing import Dict, List, Tuple


class CommitPredictor:
    """Small dense network with 3 output heads (commit, files, crossref)."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, lr: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_commit = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_commit = np.zeros(1, dtype=np.float32)
        self.W_files = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_files = np.zeros(1, dtype=np.float32)
        self.W_crossref = np.random.randn(hidden_dim, 1).astype(np.float32) * scale2
        self.b_crossref = np.zeros(1, dtype=np.float32)
        self.losses: List[float] = []

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._X = X
        self._Z1 = X @ self.W1 + self.b1
        self._A1 = self.relu(self._Z1)
        commit = self.sigmoid(self._A1 @ self.W_commit + self.b_commit)
        files = self.sigmoid(self._A1 @ self.W_files + self.b_files)
        crossref = self.sigmoid(self._A1 @ self.W_crossref + self.b_crossref)
        return commit, files, crossref

    def train_step(self, X, y_commit, y_files, y_crossref) -> float:
        batch = X.shape[0]
        pred_commit, pred_files, pred_crossref = self.forward(X)
        eps = 1e-7
        loss_c = -np.mean(y_commit * np.log(pred_commit + eps) + (1 - y_commit) * np.log(1 - pred_commit + eps))
        loss_f = -np.mean(y_files * np.log(pred_files + eps) + (1 - y_files) * np.log(1 - pred_files + eps))
        loss_x = -np.mean(y_crossref * np.log(pred_crossref + eps) + (1 - y_crossref) * np.log(1 - pred_crossref + eps))
        loss = loss_c + loss_f + loss_x

        d_c = (pred_commit - y_commit.reshape(-1, 1)) / batch
        d_f = (pred_files - y_files.reshape(-1, 1)) / batch
        d_x = (pred_crossref - y_crossref.reshape(-1, 1)) / batch
        dA1 = d_c @ self.W_commit.T + d_f @ self.W_files.T + d_x @ self.W_crossref.T
        dZ1 = dA1 * (self._Z1 > 0).astype(np.float32)
        dW1 = self._X.T @ dZ1

        clip = max(1.0, self.lr * 10)
        self.W1 -= self.lr * np.clip(dW1, -clip, clip)
        self.b1 -= self.lr * np.clip(dZ1.sum(axis=0), -clip, clip)
        self.W_commit -= self.lr * np.clip(self._A1.T @ d_c, -clip, clip)
        self.b_commit -= self.lr * np.clip(d_c.sum(axis=0), -clip, clip)
        self.W_files -= self.lr * np.clip(self._A1.T @ d_f, -clip, clip)
        self.b_files -= self.lr * np.clip(d_f.sum(), -clip, clip)
        self.W_crossref -= self.lr * np.clip(self._A1.T @ d_x, -clip, clip)
        self.b_crossref -= self.lr * np.clip(d_x.sum(axis=0), -clip, clip)
        return float(loss)

    def fit(self, X, y_commit, y_files, y_crossref, epochs=100, batch_size=32, verbose=True):
        n = X.shape[0]
        self.losses = []
        for epoch in range(epochs):
            perm = np.random.permutation(n)
            epoch_loss, batches = 0.0, 0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                epoch_loss += self.train_step(X[idx], y_commit[idx], y_files[idx], y_crossref[idx])
                batches += 1
            avg = epoch_loss / max(batches, 1)
            self.losses.append(avg)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg:.4f}")
        return self.losses

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        commit, files, crossref = self.forward(X)
        return {"commit_prob": commit.flatten(), "file_count": files.flatten(), "crossref_prob": crossref.flatten()}

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1, W_commit=self.W_commit, b_commit=self.b_commit,
                 W_files=self.W_files, b_files=self.b_files, W_crossref=self.W_crossref,
                 b_crossref=self.b_crossref, input_dim=np.array(self.input_dim), hidden_dim=np.array(self.hidden_dim))

    @classmethod
    def load(cls, path: str) -> "CommitPredictor":
        data = np.load(path)
        m = cls(input_dim=int(data["input_dim"]), hidden_dim=int(data["hidden_dim"]))
        for k in ("W1", "b1", "W_commit", "b_commit", "W_files", "b_files", "W_crossref", "b_crossref"):
            setattr(m, k, data[k])
        return m

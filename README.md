# commit-predictor

**Commit pattern predictor — a tiny dense network for forecasting fleet git activity.**

Pure-numpy implementation with zero hard dependencies beyond numpy. Predicts:
1. Whether a repo will have a commit in the next hour
2. How many files will be changed
3. Whether the commit will cross-reference another repo

## Installation

```bash
pip install commit-predictor
```

## Quick Start

```python
from commit_predictor import CommitPoint, train_commit_predictor

commits = [
    CommitPoint(sha="abc123", repo="my-repo", author="alice", timestamp=1700000000.0,
                message="fix bug", files_changed=3, insertions=10, deletions=5,
                is_merge=False, cross_refs=[], languages=[".py"]),
    # ... more commits
]

model, metrics = train_commit_predictor(commits, repos=["my-repo"], epochs=50)
print(metrics)

# Save/load
model.save("model.npz")
loaded = CommitPredictor.load("model.npz")
```

## License

MIT

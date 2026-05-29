# commit-predictor

Tiny dense neural network for forecasting fleet git activity — predicts commits-per-hour, files-changed, and cross-repo references from commit history. Pure NumPy, zero dependencies beyond that.

## What This Gives You

- **Commit prediction** — Will a repo have a commit in the next hour?
- **File change forecasting** — How many files will the next commit touch?
- **Cross-reference detection** — Will the commit reference another repo?
- **Pure NumPy** — No PyTorch, no TensorFlow — a ~200-line dense network you can read end-to-end
- **Save/load** — Serialize trained models as `.npz` files

## Quick Start

```python
from commit_predictor import CommitPoint, train_commit_predictor

commits = [
    CommitPoint(
        sha="abc123", repo="my-repo", author="alice", timestamp=1700000000.0,
        message="fix bug", files_changed=3, insertions=10, deletions=5,
        is_merge=False, cross_refs=[], languages=[".py"],
    ),
    # ... more commits
]

model, metrics = train_commit_predictor(commits, repos=["my-repo"], epochs=50)
print(metrics)

# Save and reload
model.save("model.npz")
loaded = CommitPredictor.load("model.npz")
```

## API Reference

### `CommitPoint`

| Field | Description |
|-------|-------------|
| `sha` | Commit hash |
| `repo` | Repository name |
| `author` | Commit author |
| `timestamp` | Unix timestamp |
| `files_changed` | Number of files modified |
| `insertions` / `deletions` | Line counts |
| `is_merge` | Whether it's a merge commit |
| `cross_refs` | Other repos referenced |
| `languages` | File extensions involved |

### `train_commit_predictor(commits, repos, epochs)`

Returns `(model, metrics)` — trained model and training metrics dict.

### `CommitPredictor`

```python
model.predict(features)    # Predict commit activity
model.save(path)           # Save to .npz
CommitPredictor.load(path) # Load from .npz
```

## How It Fits

- **[fleet-cicd-agent](https://github.com/SuperInstance/fleet-cicd-agent)** — Deployment scheduling informed by predicted commit patterns
- **[co-captain-git-agent](https://github.com/SuperInstance/co-captain-git-agent)** — Fleet dispatch uses predictions to anticipate workload
- **[cocapn-explain-rs](https://github.com/SuperInstance/cocapn-explain-rs)** — Explain which features drive commit predictions
- **[ccc-os](https://github.com/SuperInstance/ccc-os)** — Fleet monitoring incorporates commit predictions into status reports

## Testing

56 tests covering feature extraction, model training, prediction, save/load round-trips, and edge cases.

```bash
pip install -e ".[dev]"
pytest
```

## Installation

```bash
pip install commit-predictor
```

Or from source:

```bash
git clone https://github.com/SuperInstance/commit-predictor.git
cd commit-predictor
pip install -e .
```

Requires Python 3.11+, NumPy ≥ 1.24.

## License

MIT

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance) ecosystem.

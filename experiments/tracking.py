"""Experiment tracking (plan.md section 26).

Every run has a unique experiment ID and machine-readable JSON results, e.g.::

    gemma1b_q_v001
    gemma1b_q_v002_learned_perm
    gemma1b_q_triton_v001

Results are written to ``experiments/results/<experiment_id>.json`` by default.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "ExperimentResult",
    "new_experiment_id",
    "git_commit",
    "save_result",
    "load_result",
    "DEFAULT_RESULTS_DIR",
]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def new_experiment_id(prefix: str, version: int = 1, suffix: str = "") -> str:
    """Build an experiment ID such as ``gemma1b_q_v001_learned_perm``."""
    exp_id = f"{prefix}_v{version:03d}"
    if suffix:
        exp_id = f"{exp_id}_{suffix}"
    return exp_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentResult:
    """A single machine-readable experiment result."""

    experiment_id: str
    model: str = ""
    layer: str = ""
    method: str = ""
    status: str = "completed"
    calibration_samples: int | None = None
    metrics: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    git_commit: str | None = None
    timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> Path:
        return save_result(self, results_dir)


def save_result(
    result: ExperimentResult, results_dir: str | Path = DEFAULT_RESULTS_DIR
) -> Path:
    """Write ``result`` as JSON and return the path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if result.git_commit is None:
        result.git_commit = git_commit()
    path = results_dir / f"{result.experiment_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return path


def load_result(path: str | Path) -> ExperimentResult:
    """Load an :class:`ExperimentResult` from a JSON file."""
    data = json.loads(Path(path).read_text())
    known = {f for f in ExperimentResult.__dataclass_fields__}
    return ExperimentResult(**{k: v for k, v in data.items() if k in known})

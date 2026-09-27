"""Experiment tracking and JSON result storage (section 26)."""

import json

from experiments.tracking import (
    ExperimentResult,
    git_commit,
    load_result,
    new_experiment_id,
    save_result,
)


def test_new_experiment_id_formats():
    assert new_experiment_id("gemma1b_q") == "gemma1b_q_v001"
    assert new_experiment_id("gemma1b_q", version=2) == "gemma1b_q_v002"
    assert (
        new_experiment_id("gemma1b_q", version=2, suffix="learned_perm")
        == "gemma1b_q_v002_learned_perm"
    )


def test_save_and_load_roundtrip(tmp_path):
    result = ExperimentResult(
        experiment_id="gemma1b_q_v001",
        model="gemma-3-1b",
        layer="model.layers.0.q_proj",
        method="projection",
        calibration_samples=1024,
        metrics={"mse": 0.01, "cosine": 0.99},
        config={"grouping": "contiguous"},
    )
    path = save_result(result, tmp_path)
    assert path.exists()
    assert path.name == "gemma1b_q_v001.json"

    loaded = load_result(path)
    assert loaded.experiment_id == result.experiment_id
    assert loaded.metrics["mse"] == 0.01
    assert loaded.calibration_samples == 1024


def test_saved_json_is_machine_readable(tmp_path):
    result = ExperimentResult(experiment_id="q_v001", metrics={"ppl": 12.5})
    path = result.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["experiment_id"] == "q_v001"
    assert data["metrics"]["ppl"] == 12.5
    assert "timestamp" in data


def test_git_commit_returns_str_or_none():
    commit = git_commit()
    assert commit is None or isinstance(commit, str)

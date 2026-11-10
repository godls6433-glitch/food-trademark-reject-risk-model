"""가상 자료로 색인 → 특징 → 학습 → 동결 → 예측 → 사후 채점을 끝까지 돌린다."""

import numpy as np
import pandas as pd
import pytest

from trademark_risk import evaluate, feature_table, freeze, model, search_index, train
from trademark_risk.predict import Predictor


@pytest.fixture(scope="module")
def trained(workspace, monkeypatch_module):
    monkeypatch_module.setattr(search_index, "NameAnalyzer", _TinyAnalyzer)
    search_index.build(workspace)
    feature_table.build(workspace, workers=1)
    development = train.run(workspace)
    return workspace, development


class _TinyAnalyzer:
    @staticmethod
    def from_files(*_):
        from trademark_risk.phonetic import NameAnalyzer

        return NameAnalyzer(cmudict={}, generic_terms=["커피", "카페"])


@pytest.fixture(scope="module")
def monkeypatch_module():
    patcher = pytest.MonkeyPatch()
    yield patcher
    patcher.undo()


def test_repaired_publication_date_makes_a_mark_searchable(trained):
    paths, _ = trained
    search = search_index.load(paths)
    registry = pd.read_csv(paths.date_registry, dtype=str)
    assert registry.loc[0, "app"] in search.base.apps


def test_features_cover_the_cohort_with_clean_leakage_audit(trained):
    paths, _ = trained
    features = pd.read_parquet(paths.features)
    assert features["app"].tolist() == pd.read_parquet(paths.cohort)["app"].tolist()
    assert features[feature_table.AUDIT_COLUMNS].to_numpy().sum() == 0
    assert set(model.SEARCH_COLUMNS) <= set(features.columns)


def test_model_learns_the_planted_signal(trained):
    _, development = trained
    assert development["roc_auc"] > 0.9  # 선행상표와 겹치면 거절이 되도록 만든 자료


def test_training_and_serving_paths_agree(trained):
    paths, _ = trained
    assert train.check_serving_parity(paths) < 1e-7


def test_prediction_reports_probability_and_evidence(trained):
    paths, _ = trained
    case = pd.read_parquet(paths.cohort).query("period == 'test' and y == 1").iloc[0].to_dict()
    result = Predictor(paths).predict(case, details=3)
    assert 0 < result["rejection_probability"] < 1
    assert result["candidates"] and result["candidates"][0]["publication_date"] < case["date"]
    assert all(item["application"] != case["app"] for item in result["candidates"])


def test_frozen_model_refuses_retraining_and_detects_changes(trained):
    paths, _ = trained
    with pytest.raises(freeze.FrozenModelError):
        train.run(paths)
    original = paths.date_registry.read_bytes()
    try:
        paths.date_registry.write_bytes(original + b"\n")
        with pytest.raises(freeze.FrozenModelError):
            freeze.verify(paths)
    finally:
        paths.date_registry.write_bytes(original)
    freeze.verify(paths)


def test_retrospective_evaluation_runs_once(trained):
    paths, _ = trained
    cohort = pd.read_parquet(paths.cohort)
    test = cohort[cohort["period"].eq("test")]
    baseline = pd.DataFrame({"baseline": np.full(len(test), 0.5) + np.linspace(0, 0.01, len(test))}, index=test["app"])
    rows = evaluate.run(paths, baselines={"test": baseline})
    assert {row["period"] for row in rows} == {"test", "time_2025", "time_2026"}
    assert all(row["use"] == "EXPOSED_DEVELOPMENT_ONLY" for row in rows)
    with pytest.raises(RuntimeError):
        evaluate.run(paths)


def test_paired_bootstrap_interval_brackets_a_real_improvement():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    good, noise = y + rng.normal(0, 0.8, 600), rng.normal(0, 1, 600)
    result = evaluate.paired_group_bootstrap(np.arange(600), y, noise, good, replicates=200)
    assert result["delta_ci"][0] > 0 and result["roc_ci"][0] < result["roc_ci"][1]

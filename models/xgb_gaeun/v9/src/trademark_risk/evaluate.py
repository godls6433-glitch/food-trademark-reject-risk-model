"""동결한 모델을 기간별로 채점한다.

2024·2025·2026년 자료는 이전 버전들을 만들면서 이미 여러 번 들여다봤다. 그래서 여기서
나오는 점수는 '새 자료에서의 성능'이 아니라 **개발 자료에서의 사후 비교**이며, 결과
파일에도 그렇게 표시한다. 한 번 채점한 뒤에는 다시 고르거나 고쳐서 재지 않는다.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import freeze, model
from .config import RETROSPECTIVE_PERIODS, SEED, Paths
from .io_utils import write_json
from .text_features import text_matrix

BOOTSTRAP_REPLICATES = 500


def paired_group_bootstrap(groups, y, baseline, candidate, replicates: int = BOOTSTRAP_REPLICATES) -> dict:
    """두 모델의 ROC-AUC 차이에 대한 95% 구간.

    - 짝지은 비교: 매 반복에서 두 모델을 같은 표본으로 채점해 차이를 본다.
    - 사건군 단위 재표집: 관련 출원(같은 `source_group`)은 함께 뽑히거나 함께 빠진다.
      출원을 낱개로 뽑으면 서로 닮은 사건들이 독립인 것처럼 계산되어 구간이 좁아진다.
    """
    _, group_index = np.unique(groups, return_inverse=True)
    group_count = group_index.max() + 1
    y = np.asarray(y)
    rng = np.random.default_rng(SEED)

    results = []
    for _ in range(replicates):
        drawn = np.bincount(rng.integers(group_count, size=group_count), minlength=group_count)
        weights = drawn[group_index]
        if any(weights[y == label].sum() == 0 for label in (0, 1)):
            continue
        new = roc_auc_score(y, candidate, sample_weight=weights)
        old = roc_auc_score(y, baseline, sample_weight=weights)
        results.append([new, new - old])
    results = np.array(results)
    return dict(
        replicates=len(results),
        roc_ci=np.quantile(results[:, 0], [0.025, 0.975]).tolist(),
        delta_ci=np.quantile(results[:, 1], [0.025, 0.975]).tolist(),
    )


def score_period(cohort: pd.DataFrame, features: pd.DataFrame, bundle: dict, text_bundle: dict) -> np.ndarray:
    text = text_matrix(cohort[["name", "goods", "classes", "logo"]].to_dict("records"), text_bundle)
    search = features.loc[cohort["app"], model.SEARCH_COLUMNS].to_numpy()
    linear, tree = model.assemble(text, search, bundle["spec"], bundle["svd"])
    return model.predict_probability(bundle, linear, tree)


def run(paths: Paths, baselines: dict[str, pd.DataFrame] | None = None) -> list[dict]:
    """`baselines`: {기간: app을 색인으로 하고 이전 모델 확률을 열로 가진 표} (선택)."""
    freeze.verify(paths)
    report = paths.output / "retrospective_comparison.json"
    if report.exists():
        raise RuntimeError("Already evaluated; no reselection")

    bundle = joblib.load(paths.model)
    text_bundle = joblib.load(paths.text_bundle)
    cohort = pd.read_parquet(paths.cohort)
    features = pd.read_parquet(paths.features).set_index("app")

    rows, uncertainty = [], {}
    for period in RETROSPECTIVE_PERIODS:
        part = cohort[cohort["period"].eq(period)].reset_index(drop=True)
        part["v9"] = score_period(part, features, bundle, text_bundle)
        rows.append(dict(period=period, model="v9", use="EXPOSED_DEVELOPMENT_ONLY", **model.score(part["y"], part["v9"])))

        previous = (baselines or {}).get(period)
        if previous is not None:
            previous = previous.loc[part["app"]]
            uncertainty[period] = {}
            for name in previous.columns:
                part[name] = previous[name].to_numpy()
                rows.append(dict(period=period, model=name, use="EXPOSED_DEVELOPMENT_ONLY", **model.score(part["y"], part[name])))
                uncertainty[period][name] = paired_group_bootstrap(part["source_group"], part["y"], part[name], part["v9"])
        part.drop(columns=["goods", "logo"]).to_parquet(paths.output / f"{period}_predictions.parquet", index=False)

    write_json(report, rows)
    write_json(paths.output / "retrospective_uncertainty.json", uncertainty)
    return rows

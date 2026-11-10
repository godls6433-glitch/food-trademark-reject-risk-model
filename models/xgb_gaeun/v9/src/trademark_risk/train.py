"""고정 설정으로 한 번 학습하고 동결한다.

분할은 시간 순서다: 2017–2021 학습 → 2022 확률 보정 → 2023 개발 점검.
모델 구조·하이퍼파라미터는 이미 정해져 있어 이 단계에서 고르는 것은 없다.
"""

from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from . import freeze, model
from .config import CALIBRATION, FIT, SEED, TRAINING_PERIODS, VALIDATION, Paths
from .feature_table import AUDIT_COLUMNS
from .io_utils import read_json, write_json
from .text_features import text_matrix
from .textnorm import normalize

PARITY_SAMPLE = 5


def check_split_integrity(cohort: pd.DataFrame, excluded_apps: set[str]) -> None:
    """같은 사건군이나 같은 이름이 서로 다른 기간에 걸쳐 있으면 학습을 거부한다.

    분할출원·재출원처럼 사실상 같은 상표가 학습과 평가 양쪽에 들어가면 성능이
    부풀려지기 때문이다.
    """
    if set(cohort["app"]) & excluded_apps:
        raise ValueError("Inspected development-only cases must not enter training")
    if cohort.groupby("source_group")["period"].nunique().max() != 1:
        raise ValueError("A case group spans more than one period")
    if cohort.assign(key=cohort["name"].map(normalize)).groupby("key")["period"].nunique().max() != 1:
        raise ValueError("A normalized name spans more than one period")


def load_search_features(paths: Paths, apps: pd.Series) -> pd.DataFrame:
    features = pd.read_parquet(paths.features)
    if not features["app"].is_unique or not set(apps) <= set(features["app"]):
        raise ValueError("Search feature rows are incomplete")
    features = features.set_index("app").loc[apps]
    if features[AUDIT_COLUMNS].to_numpy().sum() != 0:
        raise ValueError("Leakage audit columns must be all zero")
    return features


def run(paths: Paths) -> dict:
    freeze.ensure_not_frozen(paths)
    started = time.time()

    cohort = pd.read_parquet(paths.cohort)
    cohort = cohort[cohort["period"].isin(TRAINING_PERIODS)].reset_index(drop=True)
    check_split_integrity(cohort, set(read_json(paths.quarantine)["applications"]))
    y = cohort["y"].to_numpy()
    fit, calibration, validation = (cohort["period"].eq(p).to_numpy() for p in (FIT, CALIBRATION, VALIDATION))

    text_bundle = joblib.load(paths.text_bundle)
    spec = model.make_spec(text_bundle)
    text = text_matrix(cohort[["name", "goods", "classes", "logo"]].to_dict("records"), text_bundle)
    svd = joblib.load(paths.text_svd) if paths.text_svd.exists() else model.fit_text_svd(text[fit, : spec["offset"]])
    search = load_search_features(paths, cohort["app"])[model.SEARCH_COLUMNS].to_numpy()

    linear, tree = model.assemble(text, search, spec, svd)
    bundle = model.fit_bundle(linear, tree, y, fit, calibration, spec, svd)
    joblib.dump(bundle, paths.model)

    probability = model.predict_probability(bundle, linear, tree)
    development = model.score(y[validation], probability[validation])
    write_json(paths.output / "development_2023.json", dict(purpose="DEVELOPMENT_NOT_FRESH", **development))

    # 학습 때 계산한 확률을 몇 건 남겨 두고, 예측 경로(predict.py)가 같은 값을 내는지 확인한다.
    sample = cohort[validation].sample(PARITY_SAMPLE, random_state=SEED)
    sample.to_parquet(paths.output / "parity_queries.parquet", index=False)
    write_json(paths.output / "parity_predictions.json", probability[sample.index].tolist())

    freeze.freeze(
        paths,
        rows=dict(fit=int(fit.sum()), calibration=int(calibration.sum()), development=int(validation.sum())),
        seed=SEED,
        seconds=round(time.time() - started),
        fresh_scored=False,
    )
    return development


def check_serving_parity(paths: Paths, tolerance: float = 1e-7) -> float:
    """학습 경로와 예측 경로의 확률 차이(최대 절댓값)."""
    from .predict import Predictor

    predictor = Predictor(paths)
    queries = pd.read_parquet(paths.output / "parity_queries.parquet").to_dict("records")
    saved = np.array(read_json(paths.output / "parity_predictions.json"))
    actual = np.array([predictor.predict(query)["rejection_probability"] for query in queries])
    difference = float(np.max(np.abs(actual - saved)))
    if difference >= tolerance:
        raise RuntimeError(f"Training/serving mismatch: {difference}")
    return difference

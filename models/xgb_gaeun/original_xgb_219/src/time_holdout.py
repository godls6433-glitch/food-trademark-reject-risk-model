"""Fit through filing year 2022; predict 2023/24 against a frozen 2022 corpus.

Run with the parent project's run.sh. No existing trained weights are loaded.
Outcome labels are the supplied 2026 snapshot, not an as-of-2022 reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import HistoricalIndex, record, static_features
from train import PARAMS, calibrator, logit, metrics, xgb

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts"
DEFAULT_OUTPUT = ROOT / "time_holdout_2022_2023_2024"
CUTOFF = "2022-12-31"
CUTOFF_DAY = int(np.datetime64(CUTOFF, "D").astype(int))
APP = "출원(국제등록)번호"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_frozen_features(reference, test, feature_names, output):
    frozen = reference.loc[reference.day <= CUTOFF_DAY].sort_values(
        ["day", "source_key"], kind="stable"
    )
    assert len(frozen) == 472936
    index = HistoricalIndex()
    start = time.monotonic()
    for r in frozen.itertuples(index=False, name=None):
        index.add(record(r[1], r[2], r[3], r[7], r[8], r[6]))
    print(f"REFERENCE frozen={len(frozen)} elapsed={time.monotonic()-start:.1f}s", flush=True)

    queries = reference.set_index("source_key", drop=False).loc["pool:" + test["순번"]]
    assert queries[APP].tolist() == test["출원번호"].tolist()
    assert queries["상표명칭"].tolist() == test["상표명칭"].tolist()
    assert (queries.day > CUTOFF_DAY).all()
    features = []
    neighbor_count = 0
    with (output / "frozen_neighbors_audit.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["순번", "rank", "prior_key", "similarity", "query_day", "prior_day"])
        for i, r in enumerate(queries.itertuples(index=False, name=None)):
            row = record(r[1], r[2], r[3], r[7], r[8], r[6])
            values = static_features(row["name"], row["classes"], row["logo"])
            historical, top = index.query(row)
            values.update(historical)
            features.append(values)
            for rank, (similarity, prior) in enumerate(top, 1):
                assert prior["day"] <= CUTOFF_DAY < row["day"]
                writer.writerow([r[0], rank, prior["key"], similarity, row["day"], prior["day"]])
                neighbor_count += 1
            if (i+1) % 10000 == 0:
                print(f"FEATURES {i+1}/{len(test)} elapsed={time.monotonic()-start:.1f}s", flush=True)
    X = pd.DataFrame(features, dtype=np.float32)
    assert X.columns.tolist() == feature_names
    assert np.isfinite(X.drop(columns="has_logo").to_numpy()).all()
    X.to_parquet(output / "test_features_frozen_2022.parquet", index=False)
    return X, {
        "reference_rows": len(frozen),
        "reference_max_date": str(np.datetime64(index.max_day, "D")),
        "test_neighbor_rows": neighbor_count,
        "all_reference_rows_on_or_before_cutoff": True,
        "all_test_neighbors_strictly_before_query": True,
        "test_reference_corpus_updated_during_2023_2024": False,
    }


def run(output):
    # Verify the cached training features and their source code before reuse.
    original_config = json.loads((ROOT / "model_config.json").read_text())
    for p in [INPUT / "cohort.parquet", INPUT / "features.parquet", ROOT / "src/features.py", ROOT / "src/train.py"]:
        assert sha(p) == original_config["training_inputs_sha256"][p.name], f"Changed input: {p}"
    cohort = pd.read_parquet(INPUT / "cohort.parquet")
    X_all = pd.read_parquet(INPUT / "features.parquet")
    year = pd.to_datetime(cohort["출원일자"]).dt.year
    train_mask, test_mask = year.between(2017, 2022), year.between(2023, 2024)
    train = cohort.loc[train_mask].reset_index(drop=True)
    # Never pass held-out outcomes into feature construction, fitting or calibration.
    test = cohort.loc[test_mask].drop(columns="registered").reset_index(drop=True)
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    assert len(train) == 191810 and len(test) == 65590
    assert (train_mask | test_mask).all() and not (train_mask & test_mask).any()
    assert set(train["순번"]).isdisjoint(test["순번"])
    assert set(train["출원번호"]).isdisjoint(test["출원번호"])
    y_train = (1 - train.registered).to_numpy(dtype=np.int8)
    folds = train.fold.to_numpy()
    assert set(folds) == {1, 2, 3, 4, 5}

    output.mkdir(parents=True, exist_ok=False)
    models = output / "models"
    models.mkdir()
    config = {
        "model": "XGB_filing_year_train_through_2022_frozen_reference",
        "target": "reject = 1 - registered",
        "train_filing_dates": ["2017-01-01", CUTOFF],
        "test_filing_dates": ["2023-01-01", "2024-12-31"],
        "train_rows": len(train), "test_rows": len(test),
        "parameters": PARAMS,
        "tuning": "none; reuse previously fixed hyperparameters, no temporal-test tuning",
        "prior_reference_cutoff": CUTOFF,
        "train_features": "verified cache: each training row uses only strictly earlier filings",
        "test_features": "newly computed from a fixed pre-2023 reference corpus",
        "feature_count": X_train.shape[1], "features": X_train.columns.tolist(),
        "calibration": "sigmoid on pre-2023 raw OOF probabilities using supplied folds 1..5",
        "final_model": "fresh model fitted on all 191810 pre-2023 labeled rows",
        "prior_model_weights_used": False,
        "test_labels_used_in_training_or_calibration": False,
        "fold_column_in_submission": "original label_folds metadata; these predictions are temporal holdout, not OOF",
        "outcome_availability_caveat": "Labels are the supplied 2026 snapshot. Filing-date split does not establish that outcomes were known by 2022-12-31.",
        "feature_design_caveat": "Feature definitions and parameters are reused from the existing independent experiment; they were not chosen using this temporal holdout's scores. This is a new chronological split of previously analyzed data, not a newly collected blind dataset.",
        "xgboost_version": xgb.__version__,
        "input_sha256": {p.name: sha(p) for p in [
            INPUT / "cohort.parquet", INPUT / "features.parquet", INPUT / "reference_safe.parquet",
            ROOT / "src/features.py", ROOT / "src/train.py", Path(__file__),
        ]},
        "pr_auc_definition": "average_precision_score; trapezoidal PR AUC also reported",
    }
    write_json(output / "model_config.json", config)
    train.to_parquet(output / "train_cohort.parquet", index=False)
    test.to_parquet(output / "test_inputs.parquet", index=False)

    # Audit cached training neighbors; exact source hashes above bind their feature code.
    train_neighbors = pd.read_csv(INPUT / "neighbors_internal.csv", dtype={"순번": str})
    train_neighbors = train_neighbors.loc[train_neighbors["순번"].isin(train["순번"])]
    assert (train_neighbors.prior_day < train_neighbors.query_day).all()
    assert (train_neighbors.query_day <= CUTOFF_DAY).all()
    reference = pd.read_parquet(INPUT / "reference_safe.parquet")
    X_test, reference_audit = make_frozen_features(reference, test, X_train.columns.tolist(), output)
    # The name/class/logo features are deterministic and independent of the corpus.
    static_cols = list(static_features("", (), 0.0))
    np.testing.assert_allclose(
        X_test[static_cols], X_all.loc[test_mask, static_cols].reset_index(drop=True),
        atol=0, rtol=0, equal_nan=True,
    )
    del reference, X_all, train_neighbors

    raw_train_oof = np.full(len(train), np.nan, dtype=np.float64)
    calibration_splits = []
    for k in range(1, 6):
        fit_mask, calibration_mask = folds != k, folds == k
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_train.loc[fit_mask], y_train[fit_mask])
        model.save_model(models / f"calibration_fold_{k}.ubj")
        raw_train_oof[calibration_mask] = model.predict_proba(X_train.loc[calibration_mask])[:, 1]
        calibration_splits.append({
            "held_out_fold": k, "training_folds": sorted(set(range(1, 6)) - {k}),
            "training_rows": int(fit_mask.sum()), "calibration_rows": int(calibration_mask.sum()),
            "maximum_filing_year": 2022,
        })
        print(f"CALIBRATION fold={k}/5 complete", flush=True)
    assert np.isfinite(raw_train_oof).all()
    np.save(output / "train_raw_oof.npy", raw_train_oof)
    cal = calibrator(raw_train_oof, y_train)
    joblib.dump(cal, models / "sigmoid_pre2023.joblib")
    write_json(output / "calibration_splits.json", calibration_splits)

    model = xgb.XGBClassifier(**PARAMS)
    model.fit(X_train, y_train)
    model.save_model(models / "xgb_train_through_2022.ubj")
    raw_test = model.predict_proba(X_test)[:, 1].astype(np.float64)
    probability = cal.predict_proba(logit(raw_test))[:, 1]
    assert np.isfinite(probability).all() and ((0 <= probability) & (probability <= 1)).all()
    np.save(output / "test_raw_probability.npy", raw_test)
    submission = test[["순번", "출원번호", "fold"]].copy()
    submission["거절확률"] = probability
    prediction_path = output / "XGB_train_to_2022_predictions_2023_2024_65590.csv"
    submission.to_csv(prediction_path, index=False, encoding="utf-8-sig", float_format="%.12f")
    # Persist the model and predictions before accessing held-out outcome labels.
    frozen_model_hashes = {p.name: sha(p) for p in [
        models / "xgb_train_through_2022.ubj", models / "sigmoid_pre2023.joblib", prediction_path,
    ]}
    print("PREDICTIONS saved; now evaluating the held-out labels", flush=True)

    y_test = (1 - cohort.loc[test_mask, "registered"]).to_numpy(dtype=np.int8)
    test_years = pd.to_datetime(test["출원일자"]).dt.year.to_numpy()
    scores = []
    for period, mask in [("2023~2024", np.ones(len(test), dtype=bool)),
                         ("2023", test_years == 2023), ("2024", test_years == 2024)]:
        scores.append({
            "period": period, "rows": int(mask.sum()), "rejected": int(y_test[mask].sum()),
            "rejection_rate": float(y_test[mask].mean()),
            "mean_predicted_probability": float(probability[mask].mean()),
            **metrics(y_test[mask], probability[mask]),
        })
    pd.DataFrame(scores).to_csv(output / "metrics_by_year.csv", index=False, encoding="utf-8-sig")
    summary = {
        "calibrated_temporal_holdout": scores,
        "raw_temporal_holdout_diagnostic": metrics(y_test, raw_test),
        "train_rows": len(train), "train_rejected": int(y_train.sum()),
        "train_rejection_rate": float(y_train.mean()),
        "constant_train_prevalence_test_brier": float(np.mean((y_train.mean()-y_test)**2)),
        "calibration_slope": float(cal.coef_[0, 0]),
        "calibration_intercept": float(cal.intercept_[0]),
        "temporal_test_score_used_for_selection": False,
    }
    write_json(output / "metrics.json", summary)
    pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    ).to_csv(output / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # Verify serialized models against every output row, including decimal CSV rounding.
    reloaded_model = xgb.XGBClassifier()
    reloaded_model.load_model(models / "xgb_train_through_2022.ubj")
    reloaded_cal = joblib.load(models / "sigmoid_pre2023.joblib")
    replay = reloaded_cal.predict_proba(logit(
        reloaded_model.predict_proba(pd.read_parquet(output / "test_features_frozen_2022.parquet"))[:, 1].astype(np.float64)
    ))[:, 1]
    saved = pd.read_csv(prediction_path, dtype={"순번": str, "출원번호": str})
    assert saved.columns.tolist() == ["순번", "출원번호", "fold", "거절확률"]
    assert saved["순번"].is_unique
    pd.testing.assert_frame_equal(saved[["순번", "출원번호", "fold"]], submission[["순번", "출원번호", "fold"]])
    assert np.max(np.abs(saved["거절확률"].to_numpy() - replay)) < 5.1e-13
    for name, expected_hash in frozen_model_hashes.items():
        p = prediction_path if name == prediction_path.name else models / name
        assert sha(p) == expected_hash
    audit = {
        **reference_audit, "train_rows": len(train), "test_rows": len(test),
        "test_rows_by_year": {str(k): int(v) for k, v in pd.Series(test_years).value_counts().sort_index().items()},
        "train_test_sequences_disjoint": True, "train_test_applications_disjoint": True,
        "training_features_verified_against_original_sha256": True,
        "training_neighbors_strictly_earlier_than_query": True,
        "csv_ids_order_and_folds_match_original_labeled_subset": True,
        "missing_predictions": int(saved["거절확률"].isna().sum()),
        "saved_model_replay_max_absolute_error": float(np.max(np.abs(saved["거절확률"].to_numpy() - replay))),
        "pre_evaluation_frozen_model_and_prediction_sha256": frozen_model_hashes,
    }
    write_json(output / "validation.json", audit)
    overall = scores[0]
    report = f"""XGBoost: 2022년까지 학습, 2023~2024년 시간 분리 예측

대상 및 결과
- 학습: label_folds 중 2017~2022년 출원 191,810건. 거절 {int(y_train.sum()):,}건.
- 예측: label_folds 중 2023년 33,070건, 2024년 32,520건, 합계 65,590건.
- 전체 학습풀의 미확정·제외 상표를 추가하지 않고 기존 라벨 집합을 연도로 분리했다.
- 제출 CSV: 순번, 출원번호, fold, 거절확률. 확률은 0~1이며 0.30은 30%다.
- 원본 label_folds 순서·fold를 보존했다. fold는 식별용 메타데이터이며 이 결과는 OOF가 아닌 시간 분리 시험 예측이다.
- 출원번호는 순번으로 원본 학습풀과 연결한 정확한 문자열 번호다.

학습과 확률 보정
- XGBClassifier: 나무 600개, 깊이 6, 학습률 0.05, min_child_weight=20,
  subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=10,
  max_bin=128, tree_method=hist, random_state=42, scale_pos_weight=1.
- 기존 실험의 고정된 219개 변수와 하이퍼파라미터를 사용하고 가중치는 새로 학습했다.
- 2017~2022년 안에서 제공된 fold 1~5로 raw OOF를 만들고 이를 정답과 함께 sigmoid 보정기에 학습했다.
- 최종 XGB는 191,810건 전체로 학습했다. 2023~2024년 정답은 모델·보정기 학습, 조기 종료, 튜닝에 사용하지 않았다.
- 보정된 확률이라도 미래 기간에서 실제 비율과 일치한다고 보장할 수는 없다.

변수와 선행 상표 비교 기준
- 상표명 길이·문자 구성, 류, 로고, 고정 단어 사전, 과거 상표 유사도 등 219개 변수.
- 학습 변수는 각 학습 상표의 출원일보다 이른 상표로만 계산한 기존 캐시의 해시를 확인하여 재사용했다.
- 예측 변수는 2022-12-31까지 출원된 P·D·학습풀 472,936건으로 새로 계산했다.
- 2023~2024년 상표를 선행 비교 목록에 추가하지 않았다. 같은 류, 동일 출원번호 제외,
  문자 n-gram 후보 검색 및 RapidFuzz 유사도 상위 5개 방식은 기존 독립 모델과 같다.
- 후보 검색은 근사 검색이며 전체 선행 상표를 전수 비교한 순위가 아니다.
- 거절 사유, 법적 상태, 사후 심사 정보는 입력 변수로 사용하지 않았다. 외부 상표 데이터 추가 없음.

검증 결과: 거절=1, 등록=0
- 전체 ROC-AUC: {overall['roc_auc']:.6f}
- 전체 PR-AUC(AP): {overall['pr_auc_ap']:.6f}
- 전체 Brier: {overall['brier']:.6f}
- 연도별 상세 결과: metrics_by_year.csv. 시험 점수에 따른 모델 수정 없음.
- 65,590행의 누락·중복·번호·순서·확률 범위·저장 모델 재현을 확인했다.

해석 범위
- '2022년까지 학습'은 출원일 기준이다. 정답은 팀에서 제공한 2026년 수집 라벨이다.
  각 정답이 2022년 말 당시 이미 확정되었는지까지 재현한 실시간 백테스트는 아니다.
- 이미 분석한 동일 데이터에 새 시간 분리를 적용한 실험이며, 새로 수집한 완전 비공개 시험셋은 아니다.
- 이전 257,400행 OOF 결과와 평가 대상·분리 방법·비교 상표 범위가 다르므로 점수를 동일 조건으로 직접 비교하지 않는다.
- 이 실험에서 5개 fold는 학습 기간 내 확률 보정에만 썼다. 시험 결과의 5-fold 평균·표준편차는 산출하지 않는다.

재현
- 작업 폴더에서: sh trademark_xgb_team_20261012/run.sh -u trademark_xgb_team_20261012/src/time_holdout.py --output NEW_OUTPUT_DIRECTORY
- 새 출력 경로만 허용하여 기존 결과를 덮어쓰지 않는다.
- model_config.json, calibration_splits.json, validation.json과 모델 파일을 함께 저장했다.
"""
    (output / "결과_설명서.txt").write_text(report, encoding="utf-8")
    write_json(output / "manifest_sha256.json", {
        str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE {prediction_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args().output.resolve())

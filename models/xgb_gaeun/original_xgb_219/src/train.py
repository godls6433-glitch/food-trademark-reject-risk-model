"""Fixed XGBoost, nested fold-only sigmoid calibration; no outer-score tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# Load installed OpenMP without changing any system or existing model files.
import ctypes
for lib in [Path("/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib")]:
    if lib.exists():
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve, auc

ROOT = Path(__file__).resolve().parents[1]
OUT, MODELS = ROOT/"artifacts", ROOT/"models"
PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=0.05, min_child_weight=20,
              subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=10,
              max_bin=128, tree_method="hist", objective="binary:logistic", eval_metric="logloss",
              random_state=42, n_jobs=4, scale_pos_weight=1, verbosity=0)


def logit(p):
    p = np.clip(p, 1e-7, 1-1e-7)
    return np.log(p/(1-p)).reshape(-1, 1)


def calibrator(p, y):
    m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)
    m.fit(logit(p), y)
    assert m.coef_[0, 0] > 0, "Unexpected inverted calibration"
    return m


def metrics(y, p):
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(roc_auc=float(roc_auc_score(y, p)), pr_auc_ap=float(average_precision_score(y, p)),
                pr_auc_trapezoidal=float(auc(recall, precision)), brier=float(brier_score_loss(y, p)))


def fit_one(X, y, train_idx, test_idx, path):
    if path.exists():
        model = xgb.XGBClassifier()
        model.load_model(path)
    else:
        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        model.save_model(path)
    return model, model.predict_proba(X.iloc[test_idx])[:, 1]


def train():
    MODELS.mkdir(exist_ok=True)
    cohort, X = pd.read_parquet(OUT/"cohort.parquet"), pd.read_parquet(OUT/"features.parquet")
    assert len(cohort) == len(X) == 257400
    y = (1-cohort.registered).to_numpy()
    folds = cohort.fold.to_numpy()
    oof, raw = np.full(len(y), np.nan), np.full(len(y), np.nan)
    audit, split_assignments, feature_importances = [], [], []
    start = time.time()
    config = {"model": "XGB_team_independent_20261012", "parameters": PARAMS,
              "target": "reject = 1 - registered", "outer_folds": "supplied label_folds, unchanged",
              "calibration": "logistic sigmoid of inner out-of-fold raw probability logit; four supplied remaining folds",
              "tuning": "none; parameters fixed before outer evaluation", "feature_count": X.shape[1],
              "features": X.columns.tolist(), "xgboost_version": xgb.__version__,
              "training_inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [OUT/"cohort.parquet", OUT/"features.parquet", ROOT/"src/features.py", ROOT/"src/train.py"]},
              "pr_auc_primary_definition": "sklearn.average_precision_score; trapezoidal PR area also reported"}
    config_path = ROOT/"model_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Do not reuse models after configuration changes"
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2))
    for outer in range(1, 6):
        test_idx = np.flatnonzero(folds == outer)
        train_idx = np.flatnonzero(folds != outer)
        inner_p = np.full(len(y), np.nan)
        for inner in sorted(set(range(1, 6))-{outer}):
            fit_idx = np.flatnonzero((folds != outer) & (folds != inner))
            cal_idx = np.flatnonzero(folds == inner)
            assert not set(fit_idx).intersection(test_idx) and not set(cal_idx).intersection(test_idx)
            _, inner_p[cal_idx] = fit_one(X, y, fit_idx, cal_idx, MODELS/f"outer{outer}_inner{inner}.ubj")
            split_assignments.append(dict(outer_test_fold=outer, calibration_fold=inner,
                                           model_training_folds=sorted(set(range(1, 6))-{outer, inner}),
                                           train_rows=len(fit_idx), calibration_rows=len(cal_idx)))
            print(f"TRAIN outer={outer} inner={inner} done elapsed={time.time()-start:.1f}s", flush=True)
        assert np.isfinite(inner_p[train_idx]).all() and np.isnan(inner_p[test_idx]).all()
        cal = calibrator(inner_p[train_idx], y[train_idx])
        joblib.dump(cal, MODELS/f"outer{outer}_sigmoid.joblib")
        model, raw[test_idx] = fit_one(X, y, train_idx, test_idx, MODELS/f"outer{outer}.ubj")
        oof[test_idx] = cal.predict_proba(logit(raw[test_idx]))[:, 1]
        feature_importances.append(model.feature_importances_)
        pd.DataFrame({"row_position": test_idx, "raw_probability": raw[test_idx], "calibrated_probability": oof[test_idx]}).to_parquet(OUT/f"fold_{outer}_predictions.parquet", index=False)
        audit.append(dict(fold=outer, rows=len(test_idx), reject_rate=float(y[test_idx].mean()),
                          calibration_slope=float(cal.coef_[0, 0]), calibration_intercept=float(cal.intercept_[0]),
                          **metrics(y[test_idx], oof[test_idx])))
        # Deliberately do not inspect/evaluate fold metrics to change the model.
        print(f"OUTER {outer}/5 completed elapsed={time.time()-start:.1f}s", flush=True)
    assert np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()
    submission = cohort[["순번", "출원번호", "fold"]].copy()
    submission["거절확률"] = oof
    submission.to_csv(OUT/"XGB_OOF_predictions_257400.csv", index=False, encoding="utf-8-sig", float_format="%.12f")
    np.save(OUT/"raw_oof.npy", raw)
    folds_table = pd.DataFrame(audit)
    folds_table.to_csv(OUT/"fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {"calibrated_oof": metrics(y, oof), "raw_oof_diagnostic": metrics(y, raw),
               "roc_auc_fold_mean": float(folds_table.roc_auc.mean()),
               "roc_auc_fold_std_population": float(folds_table.roc_auc.std(ddof=0)),
               "roc_auc_fold_std_sample": float(folds_table.roc_auc.std(ddof=1)),
               "rows": len(y), "rejection_rate": float(y.mean()),
               "constant_prevalence_brier": float(y.mean()*(1-y.mean())),
               "calibration_method_selected_before_evaluation": True}
    (OUT/"metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT/"split_audit.json").write_text(json.dumps(split_assignments, indent=2))
    pd.DataFrame({"feature": X.columns, "mean_gain_importance": np.mean(feature_importances, axis=0)}).sort_values("mean_gain_importance", ascending=False).to_csv(OUT/"feature_importance.csv", index=False, encoding="utf-8-sig")
    threshold_rows = []
    for t in np.arange(.05, .951, .05):
        high = oof >= t
        threshold_rows.append({"high_threshold": t, "high_count": int(high.sum()), "high_share": float(high.mean()),
                               "recall": float(y[high].sum()/y.sum()), "precision": float(y[high].mean()) if high.any() else None})
    pd.DataFrame(threshold_rows).to_csv(OUT/"risk_threshold_candidates.csv", index=False, encoding="utf-8-sig")
    grade_rows = []
    for low_cut in (.1, .15, .2, .25, .3):
        for high_cut in (.3, .4, .5, .6):
            if low_cut >= high_cut:
                continue
            for grade, mask in (("낮음", oof < low_cut), ("중간", (oof >= low_cut)&(oof < high_cut)), ("높음", oof >= high_cut)):
                grade_rows.append({"low_medium_threshold": low_cut, "medium_high_threshold": high_cut,
                                   "grade": grade, "count": int(mask.sum()), "share": float(mask.mean()),
                                   "actual_reject_rate": float(y[mask].mean()) if mask.any() else None})
    pd.DataFrame(grade_rows).to_csv(OUT/"risk_grade_candidate_rates.csv", index=False, encoding="utf-8-sig")
    bins = pd.cut(oof, bins=np.linspace(0, 1, 11), include_lowest=True)
    pd.DataFrame({"bin": bins, "probability": oof, "reject": y}).groupby("bin", observed=False).agg(count=("reject", "size"), mean_probability=("probability", "mean"), actual_reject_rate=("reject", "mean")).to_csv(OUT/"calibration_bins.csv", encoding="utf-8-sig")
    # Deployment model is separate from OOF submission. Calibration uses raw OOF.
    final_cal = calibrator(raw, y)
    joblib.dump(final_cal, MODELS/"final_sigmoid.joblib")
    final_model, _ = fit_one(X, y, np.arange(len(y)), np.array([0]), MODELS/"final_xgb.ubj")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"ALL DONE elapsed={time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    train()

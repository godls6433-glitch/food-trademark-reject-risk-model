"""Auxiliary refusal-type targets and display-only nearest prior trademarks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from train import ROOT, OUT, MODELS, xgb, metrics


def run():
    cohort, X = pd.read_parquet(OUT/"cohort.parquet"), pd.read_parquet(OUT/"features.parquet")
    data = next(p for p in ROOT.parent.iterdir() if p.name.startswith("02_"))
    reasons = pd.read_csv(data/"reject_reasons.csv", dtype=str, keep_default_na=False).set_index("출원번호")
    assert reasons.index.is_unique
    reason = cohort["출원번호"].map(reasons["거절사유"])
    known = reason.isin(["유사", "식별력", "복합"]).to_numpy()
    assert cohort.loc[known, "registered"].eq(0).all()
    folds = cohort.fold.to_numpy()
    parameters = dict(n_estimators=250, max_depth=4, learning_rate=.05, min_child_weight=10,
                      subsample=.85, colsample_bytree=.85, reg_lambda=10, reg_alpha=.1,
                      tree_method="hist", max_bin=128, random_state=42, n_jobs=4,
                      objective="binary:logistic", eval_metric="logloss")
    results, evaluations = {}, []
    for kind, codes in (("similarity", ["유사", "복합"]), ("distinctiveness", ["식별력", "복합"])):
        y = reason.isin(codes).astype(int).to_numpy()
        p = np.full(len(cohort), np.nan)
        for fold in range(1, 6):
            train_mask, test_mask = known & (folds != fold), folds == fold
            model = xgb.XGBClassifier(**parameters)
            model.fit(X.loc[train_mask], y[train_mask])
            p[test_mask] = model.predict_proba(X.loc[test_mask])[:, 1]
            model.save_model(MODELS/f"reason_{kind}_outer{fold}.ubj")
        results[kind] = p
        evaluations.append({"type": kind, "known_rows": int(known.sum()),
                            "positive_share": float(y[known].mean()), **metrics(y[known], p[known])})
        final = xgb.XGBClassifier(**parameters)
        final.fit(X.loc[known], y[known])
        final.save_model(MODELS/f"reason_{kind}_final.ubj")
        print(f"Auxiliary {kind} done", flush=True)
    auxiliary = cohort[["순번", "출원번호", "fold"]].copy()
    auxiliary["예상거절유형"] = np.where(results["similarity"] >= results["distinctiveness"], "유사", "식별력")
    auxiliary["유사유형점수"] = results["similarity"]
    auxiliary["식별력유형점수"] = results["distinctiveness"]
    auxiliary.to_csv(OUT/"XGB_OOF_expected_refusal_type.csv", index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.DataFrame(evaluations).to_csv(OUT/"reason_model_metrics.csv", index=False, encoding="utf-8-sig")
    (ROOT/"reason_model_config.json").write_text(json.dumps({"parameters": parameters,
        "targets": {"similarity": "유사 or 복합", "distinctiveness": "식별력 or 복합"},
        "population": "known actual refused cases in label cohort; 지정상품만 excluded from auxiliary fitting only",
        "known_training_rows": int(known.sum()), "calibration": "none; scores conditional on known refusal cases, not overall refusal probabilities",
        "source_sha256": hashlib.sha256((data/"reject_reasons.csv").read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))
    display = pd.read_parquet(OUT/"reference_display_only.parquet").set_index("source_key")
    apps = cohort.set_index("순번")["출원번호"]
    # Chunked join: legal status is attached only AFTER modeling, and labelled as current snapshot.
    out_path = OUT/"XGB_prior_top5.csv.gz"
    import gzip
    total = 0
    with gzip.open(out_path, "wt", encoding="utf-8-sig", newline="") as handle:
        first = True
        for part in pd.read_csv(OUT/"neighbors_internal.csv", dtype={"순번": str, "prior_key": str}, chunksize=100000):
            assert (part.prior_day < part.query_day).all()
            z = part.join(display, on="prior_key", validate="many_to_one")
            z["대상출원번호"] = z["순번"].map(apps)
            z = z.rename(columns={"rank": "유사도순위", "출원(국제등록)번호": "선행출원번호", "상표명칭": "선행상표명",
                                  "상품분류": "선행상표류", "법적상태": "법적상태_수집시점_모델미사용",
                                  "출원(국제등록)일자": "선행출원일", "similarity": "문자유사도"})
            assert z["선행출원번호"].ne(z["대상출원번호"]).all()
            z[["순번", "대상출원번호", "유사도순위", "선행출원번호", "선행상표명", "선행상표류", "선행출원일", "법적상태_수집시점_모델미사용", "문자유사도"]].to_csv(handle, index=False, header=first, float_format="%.8f")
            first = False
            total += len(z)
    print(f"Display-only prior list exported: {total} rows", flush=True)


if __name__ == "__main__":
    run()

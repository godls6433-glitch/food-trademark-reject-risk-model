from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from features import HistoricalIndex, WORDS, TOKEN_FEATURES, record, static_features

ROOT = Path(__file__).resolve().parents[1]
DATA = next(p for p in ROOT.parent.iterdir() if p.name.startswith("02_"))
OUT = ROOT / "artifacts"
APP = "출원(국제등록)번호"
DAY = "출원(국제등록)일자"
SAFE_COLUMNS = ["순번", "상표명칭", "상품분류", APP, DAY, "명칭구분"]


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def prepare():
    OUT.mkdir(exist_ok=True)
    label = read(DATA / "label_folds.csv")
    assert len(label) == 257400 and label["순번"].is_unique
    assert set(label.fold) == set("12345")
    assert set(label.registered) == {"0", "1"}
    paths = sorted(DATA.rglob("tm_pool_*.csv"))
    pool = pd.concat([read(p) for p in paths], ignore_index=True)
    assert len(pool) == 395982 and pool["순번"].is_unique
    p_paths = sorted(DATA.glob("P_*.csv"))
    prior_p = pd.concat([read(p) for p in p_paths], ignore_index=True)
    # The supplied files have already removed the nine boundary duplicates.
    assert len(prior_p) == 141494 and prior_p["순번"].is_unique
    prior_d = read(DATA / "D_2005-2016_soml.csv")
    assert len(prior_d) == 93292
    merged = label.merge(pool, on="순번", how="left", validate="one_to_one", suffixes=("_label", ""), indicator=True)
    assert merged._merge.eq("both").all()
    name_mismatch = ~merged["상표명칭_label"].eq(merged["상표명칭"])
    merged.loc[name_mismatch, ["순번", "상표명칭_label", "상표명칭"]].to_csv(OUT/"label_name_source_differences.csv", index=False, encoding="utf-8-sig")
    assert merged["상품분류_label"].eq(merged["상품분류"]).all()
    dates = pd.to_datetime(merged[DAY], format="%Y.%m.%d")
    label_dates = pd.to_datetime(merged["출원일자"], format="%Y.%m.%d")
    assert dates.eq(label_dates).all() and dates.dt.year.between(2017, 2024).all()
    expected = merged["법적상태"].map({"등록": 1, "공고": 1, "소멸": 1, "거절": 0})
    assert expected.notna().all() and expected.eq(merged.registered.astype(int)).all()
    cohort = pd.DataFrame({"순번": merged["순번"], "출원번호": merged[APP], "fold": merged.fold.astype(int),
                           "registered": merged.registered.astype(int), "상표명칭": merged["상표명칭"],
                           "상품분류": merged["상품분류"], "출원일자": dates.dt.strftime("%Y-%m-%d")})
    cohort.to_parquet(OUT / "cohort.parquet", index=False)
    safe = []
    for prefix, frame in (("P", prior_p), ("D", prior_d), ("pool", pool)):
        z = frame[SAFE_COLUMNS].copy()
        z["source_key"] = prefix + ":" + z["순번"]
        safe.append(z)
    all_rows = pd.concat(safe, ignore_index=True)
    # Explicit days avoid dependence on pandas' datetime resolution.
    all_rows["day"] = pd.to_datetime(all_rows[DAY], format="%Y.%m.%d").to_numpy(dtype="datetime64[D]").astype("int64")
    all_rows["logo"] = all_rows["명칭구분"].map(lambda x: 1. if x == "도형복합" else (np.nan if x == "" else 0.))
    all_rows.to_parquet(OUT / "reference_safe.parquet", index=False)
    # Report-only snapshot: never passed to feature extraction or risk model.
    display = pd.concat([x.assign(source=s) for s, x in (("P", prior_p), ("D", prior_d), ("pool", pool))], ignore_index=True)
    display["source_key"] = display.source + ":" + display["순번"]
    display[["source_key", APP, "상표명칭", "상품분류", "법적상태", DAY]].to_parquet(OUT/"reference_display_only.parquet", index=False)
    hashes = {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths+p_paths+[DATA/"D_2005-2016_soml.csv", DATA/"label_folds.csv"]}
    audit = {"label_rows": len(label), "pool_rows": len(pool), "P_rows": len(prior_p), "D_rows": len(prior_d),
             "rejected": int((1-cohort.registered).sum()), "rejection_rate": float((1-cohort.registered).mean()),
             "fold_counts": cohort.fold.value_counts().sort_index().to_dict(),
             "label_scientific_application_rows": int(label["출원번호"].str.contains("[Ee][+]", regex=True).sum()),
             "class_date_label_matches": True, "label_status_matches": True,
             "label_name_source_differences": int(name_mismatch.sum()),
             "name_policy": "always original pool name, joined by sequence; keep all labels/folds",
             "exact_source_hashes_sha256": hashes,
             "permitted_feature_columns": SAFE_COLUMNS,
             "prior_status_used": False, "applicant_used": False, "external_trademark_data": False}
    (OUT/"data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (ROOT/"dictionary.json").write_text(json.dumps(WORDS, ensure_ascii=False, indent=2))
    # No targets used below. Same-day records are all queried before insertion.
    index = HistoricalIndex()
    target_positions = dict(zip("pool:" + cohort["순번"], range(len(cohort))))
    features = [None] * len(cohort)
    all_rows = all_rows.sort_values(["day", "source_key"], kind="stable")
    start, done = time.time(), 0
    maxday = int(pd.to_datetime(cohort["출원일자"]).to_numpy(dtype="datetime64[D]").astype("int64").max())
    neighbors_path = OUT/"neighbors_internal.csv"
    with neighbors_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["순번", "rank", "prior_key", "similarity", "query_day", "prior_day"])
        for day, group in all_rows.loc[all_rows.day <= maxday].groupby("day", sort=True):
            records = [record(r[1], r[2], r[3], day, r[8], r[6]) for r in group.itertuples(index=False, name=None)]
            for row in records:
                pos = target_positions.get(row["key"])
                if pos is None:
                    continue
                f = static_features(row["name"], row["classes"], row["logo"])
                history, top = index.query(row)
                f.update(history)
                features[pos] = f
                for rank, (sim, prior) in enumerate(top, 1):
                    writer.writerow([cohort.iloc[pos]["순번"], rank, prior["key"], sim, day, prior["day"]])
                done += 1
                if done % 10000 == 0:
                    print(f"FEATURES {done}/{len(cohort)} elapsed={time.time()-start:.1f}s", flush=True)
            for row in records:
                index.add(row)
    assert done == len(cohort) and all(f is not None for f in features)
    frame = pd.DataFrame(features, dtype=np.float32)
    frame.to_parquet(OUT/"features.parquet", index=False)
    assert np.isfinite(frame.drop(columns="has_logo").to_numpy()).all()
    descriptions = []
    for c in frame.columns:
        token = TOKEN_FEATURES[int(c[5:])] if c.startswith("word_") else ""
        descriptions.append({"feature": c, "literal_token": token,
                             "source": "historical_name_and_class_before_filing" if c.startswith(("prior_", "history_", "exact_", "retrieved_", "gram_")) else "name_class_logo_at_filing"})
    pd.DataFrame(descriptions).to_csv(OUT/"feature_list.csv", index=False, encoding="utf-8-sig")
    print(f"DONE features={frame.shape} seconds={time.time()-start:.1f}", flush=True)


if __name__ == "__main__":
    prepare()

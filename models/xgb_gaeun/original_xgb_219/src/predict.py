"""New-query inference. Historical evaluation comes only from the OOF CSV."""
import argparse
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from features import HistoricalIndex, record, static_features
from train import ROOT, OUT, MODELS, xgb, logit


def build_query(name, classes, logo, as_of="2026-10-12", application=""):
    cutoff = int(np.datetime64(as_of, "D").astype(int))
    ref = pd.read_parquet(OUT/"reference_safe.parquet")
    index = HistoricalIndex()
    for r in ref.loc[ref.day < cutoff].sort_values(["day", "source_key"], kind="stable").itertuples(index=False, name=None):
        index.add(record(r[1], r[2], r[3], r[7], r[8], r[6]))
    query = record(name, classes, application, cutoff, logo, "query")
    f = static_features(name, query["classes"], logo)
    h, top = index.query(query)
    f.update(h)
    cols = json.loads((ROOT/"model_config.json").read_text())["features"]
    X = pd.DataFrame([f], columns=cols, dtype=np.float32)
    return X, top, query


def predict(name, classes, logo, as_of="2026-10-12", applicant=None):
    X, top, query = build_query(name, classes, logo, as_of)
    model = xgb.XGBClassifier()
    model.load_model(MODELS/"final_xgb.ubj")
    cal = joblib.load(MODELS/"final_sigmoid.joblib")
    p = float(cal.predict_proba(logit(model.predict_proba(X)[:, 1].astype(np.float64)))[0, 1])
    kind = {}
    for code, title in (("similarity", "유사"), ("distinctiveness", "식별력")):
        reason = xgb.XGBClassifier()
        reason.load_model(MODELS/f"reason_{code}_final.ubj")
        kind[title] = float(reason.predict_proba(X)[0, 1])
    display = pd.read_parquet(OUT/"reference_display_only.parquet").set_index("source_key")
    neighbors = []
    for score, row in top:
        item = display.loc[row["key"]]
        neighbors.append({"상표명": row["name"], "류": list(row["classes"]), "출원번호": row["application"],
                          "법적상태_수집시점": item["법적상태"], "문자유사도": score,
                          "출원일": item["출원(국제등록)일자"]})
    return {"상표명": name, "류": list(query["classes"]), "로고유무": bool(logo),
            "출원인_입력값_모델미사용": applicant, "조회기준일": as_of, "거절확률": p, "거절확률_percent": round(100*p, 2),
            "예상거절유형": max(kind, key=kind.get), "거절유형_조건부점수": kind,
            "위험등급": "팀의 경계값 결정 전", "비슷한선행상표": neighbors,
            "참고": "근사 후보 검색 후 문자유사도 순. 상태는 수집 당시 정보이며 출원 당시 상태를 뜻하지 않음. 새 예측 모델은 전체 라벨로 학습했으므로 과거 사례 재입력은 OOF 평가가 아님."}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--classes", required=True, help="29|30|32|43 중 선택")
    p.add_argument("--logo", required=True, type=int, choices=[0, 1])
    p.add_argument("--as-of", default="2026-10-12")
    p.add_argument("--applicant", default=None)
    args = p.parse_args()
    print(json.dumps(predict(args.name, args.classes, args.logo, args.as_of, args.applicant), ensure_ascii=False, indent=2))

"""정리한 코드가 연구 작업 폴더의 동결 모델(v9)과 같은 값을 내는지 실제 자료로 확인한다.

    python scripts/check_against_frozen_v9.py            # 대조만
    python scripts/check_against_frozen_v9.py --adopt    # 통과하면 동결 모델을 artifacts/로 가져와 predict에서 쓸 수 있게 한다

확인하는 것
1. 색인: 이 저장소 코드로 선행상표 풀을 다시 색인했을 때 행 수.
2. 검색 특징: 표본 출원의 32개 특징이 v9가 저장해 둔 값과 일치하는지.
3. 예측: 2024년 전체 행의 확률이 v9가 저장해 둔 예측과 일치하는지.
4. 학습·예측 경로 일치: 학습 때 남겨 둔 5건을 색인부터 다시 계산해 같은 확률이 나오는지.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trademark_risk import freeze, model, search_index  # noqa: E402
from trademark_risk.config import QUERY_FIELDS, Paths  # noqa: E402
from trademark_risk.evaluate import score_period  # noqa: E402
from trademark_risk.feature_table import search_features  # noqa: E402
from trademark_risk.io_utils import read_json  # noqa: E402

TOLERANCE = 1e-6  # 서로 다른 OS·CPU에서 생기는 부동소수점 차이는 허용


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=250, help="기간마다 다시 계산해 볼 출원 수")
    parser.add_argument("--adopt", action="store_true")
    args = parser.parse_args()

    paths = Paths.default()
    v9 = paths.workspace / "trademark_risk_v9"
    paths.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(v9 / "data/검증된_공고일_보완.csv", paths.date_registry)

    search, audit = search_index.build_in_memory(paths)
    print(f"[1] indexed prior marks: {audit['indexed_rows']:,} (repaired dates: {audit['repaired_applications']})")

    cohort = pd.read_parquet(paths.cohort)
    stored = pd.read_parquet(v9 / "artifacts/features.parquet").set_index("app")
    parity = pd.read_parquet(v9 / "artifacts/parity_queries.parquet")
    sample = pd.concat([part.sample(min(args.sample, len(part)), random_state=7) for _, part in cohort.groupby("period")])
    sample = pd.concat([sample, parity[cohort.columns]]).drop_duplicates("app")
    recomputed = pd.DataFrame([search_features(search, q) for q in sample[list(QUERY_FIELDS)].to_dict("records")]).set_index("app")
    columns = model.SEARCH_COLUMNS
    feature_gap = np.abs(recomputed[columns].to_numpy(float) - stored.loc[recomputed.index, columns].to_numpy(float)).max()
    print(f"[2] search features, {len(sample):,} applications: max |difference| = {feature_gap:.2e}")

    bundle = joblib.load(v9 / "artifacts/candidate.joblib")
    text_bundle = joblib.load(paths.text_bundle)
    test = cohort[cohort["period"].eq("test")].reset_index(drop=True)
    saved = pd.read_parquet(v9 / "artifacts/test_predictions.parquet").set_index("app").loc[test["app"], "v9"].to_numpy()
    probability = score_period(test, stored, bundle, text_bundle)
    prediction_gap = np.abs(probability - saved).max()
    print(f"[3] 2024 predictions, {len(test):,} applications: max |difference| = {prediction_gap:.2e}")
    print(f"    ROC-AUC {model.score(test['y'], probability)['roc_auc']:.6f}")

    parity_saved = np.array(read_json(v9 / "artifacts/parity_predictions.json"))
    parity_gap = np.abs(score_period(parity, recomputed, bundle, text_bundle) - parity_saved).max()
    print(f"[4] index → features → model on the 5 parity cases: max |difference| = {parity_gap:.2e}")

    if max(feature_gap, prediction_gap, parity_gap) > TOLERANCE:
        raise SystemExit("MISMATCH: the refactored code does not reproduce frozen v9")
    print("OK: refactored code reproduces frozen v9")

    if args.adopt:
        freeze.ensure_not_frozen(paths)
        joblib.dump(search, paths.search_index)
        shutil.copyfile(v9 / "artifacts/candidate.joblib", paths.model)
        for name in ("features.parquet", "parity_queries.parquet", "parity_predictions.json"):
            shutil.copyfile(v9 / "artifacts" / name, paths.output / name)
        freeze.freeze(paths, adopted_from="trademark_risk_v9/artifacts/candidate.joblib", fresh_scored=False)
        print(f"adopted frozen v9 into {paths.output}")


if __name__ == "__main__":
    main()

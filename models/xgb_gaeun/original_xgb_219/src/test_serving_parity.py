import json
import numpy as np
import pandas as pd
from predict import build_query, OUT

cohort = pd.read_parquet(OUT/"cohort.parquet")
cached = pd.read_parquet(OUT/"features.parquet")
# Historical replay tests extraction only, never presented as model evaluation.
results = []
for pos in (0, 150000):
    r = cohort.iloc[pos]
    actual, _, _ = build_query(r["상표명칭"], r["상품분류"], float(cached.iloc[pos].has_logo), r["출원일자"], r["출원번호"])
    expected = cached.iloc[[pos]].to_numpy()
    np.testing.assert_allclose(actual.to_numpy(), expected, rtol=1e-6, atol=1e-7, equal_nan=True)
    results.append({"row_position": pos, "sequence": r["순번"], "all_219_features_match": True})
(OUT/"serving_feature_parity.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(json.dumps(results, ensure_ascii=False))

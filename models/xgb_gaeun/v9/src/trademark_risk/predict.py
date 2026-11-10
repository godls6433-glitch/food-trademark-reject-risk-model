"""출원 한 건의 거절 확률과 근거 후보를 낸다."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import freeze, model, search_index
from .config import QUERY_FIELDS, Paths
from .pair_search import PairSearch
from .text_features import text_matrix
from .textnorm import normalize, parse_date

_PLACEHOLDER_NAMES = {"", "정보없음", "상표명정보없음"}


def clean_query(raw: dict) -> dict:
    """허용 목록에 있는 입력만 남긴다.

    법적 상태·등록번호·거절 사유 같은 사후 정보가 함께 넘어와도 여기서 버려지므로
    예측에 영향을 줄 수 없다(테스트로 확인).
    """
    query = {field: str(raw.get(field, "")) for field in QUERY_FIELDS if field != "logo"}
    query["logo"] = int(raw.get("logo", 0))
    query["date"] = parse_date(query["date"])
    if query["logo"] not in (0, 1) or not query["goods"].strip() or normalize(query["name"]) in _PLACEHOLDER_NAMES:
        raise ValueError("Missing query input")
    return query


class Predictor:
    def __init__(self, paths: Paths | None = None, search: PairSearch | None = None):
        self.paths = paths or Paths.default()
        freeze.verify(self.paths)
        self.bundle = joblib.load(self.paths.model)
        self.text_bundle = joblib.load(self.paths.text_bundle)
        self.search = search or search_index.load(self.paths)
        registry = pd.read_csv(self.paths.date_registry, dtype=str, keep_default_na=False)
        self.repaired_sources = dict(zip(registry["app"], registry["source_type"]))

    def predict(self, raw_query: dict, details: int = 5) -> dict:
        query = clean_query(raw_query)
        self.search.base.filing_date(query)

        prior, _, _ = self.search.base.extract(query)
        pair, candidates, _ = self.search.extract(query, details=details)
        features = {**prior, **pair}
        for candidate in candidates:
            # 이 후보의 공고일이 원자료에 있던 것인지, v9에서 근거를 확인해 보완한 것인지.
            candidate["publication_source"] = self.repaired_sources.get(candidate["application"], "recorded_in_pool")

        search = np.array([[features[column] for column in model.SEARCH_COLUMNS]])
        linear, tree = model.assemble(text_matrix([query], self.text_bundle), search, self.bundle["spec"], self.bundle["svd"])
        probability = float(model.predict_probability(self.bundle, linear, tree)[0])

        return dict(
            application=query["app"],
            name=query["name"],
            classes=query["classes"],
            as_of=query["date"],
            rejection_probability=probability,
            review_recommended=probability >= self.bundle["review_threshold"],
            candidates=candidates,
            features=features,
            research_only=True,
            original_filing_goods_verified=False,
        )

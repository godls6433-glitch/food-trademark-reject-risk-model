"""상표명·지정상품 자체에서 나오는 특징 (v3에서 도입).

두 묶음을 만든다.

1. 문자 TF-IDF 3종 — 이름 문자 2~4gram, 이름 자모 3~5gram, 지정상품 단어.
2. 구조·문맥 수치 — 이름 길이, 한글/영문/숫자 비율, 로고 유무, 상품류, 그리고
   역할 사전(산지·원재료·상품명 등)의 단어가 이름에 들어 있는지와 그 단어가
   지정상품에도 나오는지.

TF-IDF 어휘와 IDF는 학습 기간(2017–2021) 자료에만 맞춘 것을 그대로 불러 쓴다.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .textnorm import TARGET_CLASSES, goods_terms, normalize, parse_classes, to_jamo

# 사전의 단어가 이름에서 맡는 역할. 식별력이 약한 표현을 찾기 위한 후보 분류이며
# 사전에 없다는 사실을 '식별력이 높다'는 근거로 쓰지 않는다.
ROLES = (
    "LOCATION", "INGREDIENT", "PRODUCT", "SERVICE_CONTENT", "QUALITY", "EFFECT",
    "USE", "METHOD", "PRICE", "COMMON_EXPRESSION", "REGULATED", "BRAND_CANDIDATE",
)  # fmt: skip
CLASS_ORDER = tuple(sorted(TARGET_CLASSES))

STRUCTURE_COLUMNS = (
    "length", "hangul_ratio", "latin_ratio", "digit_ratio", "logo", "class_count",
    "goods_count", "class_29", "class_30", "class_32", "class_43",
)  # fmt: skip


def _occurrences(text: str, term: str) -> set[int]:
    """`text` 안에서 `term`이 차지하는 문자 위치(겹치는 출현 포함)."""
    positions: set[int] = set()
    start = text.find(term)
    while start >= 0:
        positions.update(range(start, start + len(term)))
        start = text.find(term, start + 1)
    return positions


def structure_features(query: dict, lexicon: dict[str, list[str]]) -> dict[str, float]:
    """출원 한 건의 구조·문맥 수치 특징.

    `kb_<역할>_name`은 이름 글자 중 그 역할 단어가 덮는 비율,
    `kb_<역할>_context`는 그중 지정상품에도 등장하는 단어가 덮는 비율이다.
    문자 위치의 합집합으로 세므로 겹치는 단어를 두 번 세지 않는다.
    """
    name = normalize(query["name"])
    length = max(len(name), 1)
    classes = parse_classes(query["classes"])
    goods = sorted(goods_terms(query["goods"]))

    features = {
        "length": min(length, 100) / 100,
        "hangul_ratio": sum("가" <= c <= "힣" for c in name) / length,
        "latin_ratio": sum("a" <= c <= "z" for c in name) / length,
        "digit_ratio": sum(c.isdecimal() for c in name) / length,
        "logo": int(query["logo"]),
        "class_count": len(classes) / 45,
        "goods_count": min(len(goods), 200) / 200,
    }
    for number in CLASS_ORDER:
        features[f"class_{number}"] = int(number in classes)

    covered_any: set[int] = set()
    context_any: set[int] = set()
    for role in ROLES:
        covered: set[int] = set()
        in_context: set[int] = set()
        for term in lexicon.get(role, []):
            positions = _occurrences(name, term)
            if not positions:
                continue
            covered |= positions
            if any(term in item for item in goods):
                in_context |= positions
        covered_any |= covered
        context_any |= in_context
        features[f"kb_{role}_name"] = len(covered) / length
        features[f"kb_{role}_context"] = len(in_context) / length
        for number in CLASS_ORDER:
            features[f"kb_{role}_context_{number}"] = len(in_context) / length * int(number in classes)

    features["kb_any_name"] = len(covered_any) / length
    features["kb_any_context"] = len(context_any) / length
    features["kb_unmatched"] = 1 - len(covered_any) / length
    return features


def text_matrix(queries: list[dict], bundle: dict) -> sparse.csr_matrix:
    """출원 목록 → [TF-IDF 3종 | 구조·문맥 수치] 희소 행렬.

    `bundle`은 학습 기간에 맞춘 `vectorizers`, 역할 사전 `lexicon`, 수치 열 순서
    `numeric_columns`를 담고 있다. 앞쪽 `bundle["offset"]`개 열이 TF-IDF다.
    """
    names = [normalize(q["name"]) for q in queries]
    documents = [
        names,
        [to_jamo(name) for name in names],
        [" ".join(sorted(goods_terms(q["goods"]))) for q in queries],
    ]
    blocks = [vectorizer.transform(texts) for vectorizer, texts in zip(bundle["vectorizers"], documents)]

    rows = [structure_features(q, bundle["lexicon"]) for q in queries]
    numeric = [[row[column] for column in bundle["numeric_columns"]] for row in rows]
    blocks.append(sparse.csr_matrix(numeric, dtype=np.float64))
    return sparse.hstack(blocks, format="csr")

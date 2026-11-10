"""특징 행렬 조립과 혼합 모델 (구조는 v8에서 확정).

같은 입력을 두 갈래로 본다.

- 선형 갈래: 6만여 차원 문자 TF-IDF를 그대로 쓰는 로지스틱 회귀. 특정 글자 조각의
  효과를 직접 잡는다.
- 트리 갈래: TF-IDF를 64차원으로 압축해 수치 특징과 함께 넣은 XGBoost. 선행상표
  유사도와 상품 겹침 사이의 비선형 조합을 잡는다.

두 확률을 0.5:0.5로 섞고, 2022년 자료로 시그모이드 보정을 한 번 더 한다.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import expit, logit
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from .config import LINEAR_C, REVIEW_QUANTILE, SEED, SVD_COMPONENTS, SVD_SEED, TREE_WEIGHT, XGB_PARAMS
from .pair_search import PAIR_COLUMNS
from .prior_search import PRIOR_COLUMNS
from .text_features import ROLES, STRUCTURE_COLUMNS

# 문맥 특징은 '이름 속 단어가 지정상품에도 나온다'는 관측만 쓴다.
# 사전에 없는 이름(kb_unmatched)은 식별력의 증거가 아니므로 넣지 않는다(v7의 수정).
CONTEXT_COLUMNS = tuple(f"kb_{role}_context" for role in ROLES[:10])
SEARCH_COLUMNS = PRIOR_COLUMNS + PAIR_COLUMNS

_EPSILON = 1e-6


def make_spec(text_bundle: dict) -> dict:
    """텍스트 행렬에서 어느 열을 꺼내 쓸지 기록한다(모델과 함께 저장)."""
    offset = text_bundle["offset"]
    position = {column: offset + i for i, column in enumerate(text_bundle["numeric_columns"])}
    return dict(
        offset=offset,
        basic_selector=[position[c] for c in STRUCTURE_COLUMNS],
        context_selector=[position[c] for c in CONTEXT_COLUMNS],
        basic_names=list(STRUCTURE_COLUMNS),
        context_names=list(CONTEXT_COLUMNS),
        prior_names=list(SEARCH_COLUMNS),
    )


def fit_text_svd(tfidf_fit_rows: sparse.csr_matrix) -> TruncatedSVD:
    """TF-IDF 압축기. 반드시 학습 기간 행만 넣어 맞춘다."""
    return TruncatedSVD(n_components=SVD_COMPONENTS, n_iter=4, random_state=SVD_SEED).fit(tfidf_fit_rows)


def assemble(text: sparse.csr_matrix, search: np.ndarray, spec: dict, svd: TruncatedSVD):
    """텍스트 행렬 + 검색 특징 32개 → (선형 모델 입력, 트리 모델 입력).

    선형: [TF-IDF 62,000 | 구조 11 | 문맥 10 | 검색 32]  (희소)
    트리: [SVD 64       | 구조 11 | 문맥 10 | 검색 32]  (밀집 117열)
    """
    tfidf = text[:, : spec["offset"]]
    structure = text[:, spec["basic_selector"]].toarray()
    context = text[:, spec["context_selector"]].toarray()
    search = np.asarray(search).astype(np.float32)

    linear = sparse.hstack([tfidf, sparse.csr_matrix(np.hstack([structure, context, search]))], format="csr")
    tree = np.hstack([svd.transform(tfidf), structure, context, search]).astype(np.float32)
    return linear, tree


def _to_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(probability, _EPSILON, 1 - _EPSILON)).reshape(-1, 1)


def blend(bundle: dict, linear, tree) -> np.ndarray:
    """보정 전 혼합 점수."""
    tree_probability = bundle["tree"].predict_proba(tree)[:, 1]
    linear_probability = expit(bundle["linear"].decision_function(linear))
    return bundle["weight"] * tree_probability + (1 - bundle["weight"]) * linear_probability


def predict_probability(bundle: dict, linear, tree) -> np.ndarray:
    """보정된 거절 확률."""
    return bundle["calibrator"].predict_proba(_to_logit(blend(bundle, linear, tree)))[:, 1]


def fit_bundle(linear, tree, y: np.ndarray, fit: np.ndarray, calibration: np.ndarray, spec: dict, svd) -> dict:
    """고정 설정으로 한 번 학습한다. `fit`/`calibration`은 행 선택용 불리언 배열."""
    tree_model = XGBClassifier(**XGB_PARAMS).fit(tree[fit], y[fit])
    linear_model = LogisticRegression(C=LINEAR_C, solver="liblinear", max_iter=500, random_state=SEED)
    linear_model.fit(linear[fit], y[fit])
    if (linear_model.n_iter_ >= 500).any():
        raise RuntimeError("Linear model did not converge")

    bundle = dict(
        version="v9",
        kind="blend",
        tree=tree_model,
        linear=linear_model,
        weight=TREE_WEIGHT,
        spec=spec,
        svd=svd,
        parameters=XGB_PARAMS,
    )
    # 확률 보정(Platt scaling): 학습에 쓰지 않은 2022년 자료에서 점수→확률 대응을 맞춘다.
    raw = blend(bundle, linear, tree)
    calibrator = LogisticRegression(C=1000, max_iter=300, random_state=SEED)
    calibrator.fit(_to_logit(raw[calibration]), y[calibration])
    if calibrator.coef_[0, 0] <= 0:
        raise RuntimeError("Calibration slope must be positive")
    bundle["calibrator"] = calibrator

    # 검토 권고선: 보정 자료의 실제 거절 사건 중 80%가 이 확률 이상이 되도록 잡는다.
    calibrated = predict_probability(bundle, linear, tree)
    refused = calibrated[calibration][y[calibration] == 1]
    bundle["review_threshold"] = float(np.quantile(refused, REVIEW_QUANTILE, method="lower"))
    return bundle


def score(y, probability) -> dict:
    """순위 구분력(ROC-AUC, PR-AUC)과 확률 정확도(Brier)."""
    return dict(
        n=len(y),
        rejection_rate=float(np.mean(y)),
        roc_auc=float(roc_auc_score(y, probability)),
        pr_auc=float(average_precision_score(y, probability)),
        brier=float(brier_score_loss(y, probability)),
    )

"""경로와 고정 설정.

모델 구조와 하이퍼파라미터는 점수를 보기 전에 고정했다(v8에서 2023년 자료로 선택,
v9에서는 그대로 재사용). 그래서 탐색 범위가 아니라 상수로 둔다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 42

# 시간 순서 분할: 학습 → 확률 보정 → 개발 점검 → 사후 비교(이미 여러 번 본 자료).
FIT, CALIBRATION, VALIDATION = "fit", "calibration", "validation"
TRAINING_PERIODS = (FIT, CALIBRATION, VALIDATION)  # 2017–2021 / 2022 / 2023
RETROSPECTIVE_PERIODS = ("test", "time_2025", "time_2026")  # 2024 / 2025 / 2026

# 예측에 쓸 수 있는 입력의 전부. 법적 상태·등록번호·거절 사유는 받지 않는다.
QUERY_FIELDS = ("app", "name", "goods", "classes", "date", "logo")

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=20,
    reg_lambda=5,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=2,
    tree_method="hist",
    max_bin=256,
    random_state=SEED,
    eval_metric="logloss",
)
LINEAR_C = 0.25  # 문자 TF-IDF 로지스틱 회귀의 규제 강도
TREE_WEIGHT = 0.5  # 최종 점수 = 0.5 × XGBoost + 0.5 × 로지스틱 회귀
REVIEW_QUANTILE = 0.2  # 검토 권고선 = 보정 자료 거절 사건 확률의 하위 20% 지점(재현율 80% 목표)

SVD_COMPONENTS = 64  # 6만여 차원 TF-IDF를 트리 입력용으로 압축
SVD_SEED = 61

MAX_CANDIDATES = 512  # 검색 채널마다 정밀 비교로 넘기는 선행상표 후보 수
FEATURE_CHUNK_ROWS = 500  # 특징 계산 중간 저장 단위


@dataclass(frozen=True)
class Paths:
    """입력(작업 폴더)과 출력(이 저장소의 artifacts/) 위치.

    원자료와 이전 실험 단계의 산출물은 저장소에 올리지 않는다(용량·KIPRIS 원문).
    `TRADEMARK_WORKSPACE` 환경변수로 작업 폴더를 지정하고, 없으면 저장소의 상위 폴더를 쓴다.
    """

    workspace: Path
    output: Path

    @classmethod
    def default(cls) -> Paths:
        workspace = Path(os.environ.get("TRADEMARK_WORKSPACE", REPO_ROOT.parent))
        output = Path(os.environ.get("TRADEMARK_ARTIFACTS", REPO_ROOT / "artifacts"))
        return cls(workspace=workspace.resolve(), output=output.resolve())

    # ── 입력: 이전 단계에서 만들어 둔 자료 ─────────────────────────────────────
    @property
    def pool_tables(self) -> tuple[Path, Path]:
        """선행상표 풀. 현존 기록(v1 정리)과 소멸 기록(v2에서 추가)."""
        return (
            self.workspace / "trademark_risk_v1/artifacts/pool_raw.parquet",
            self.workspace / "trademark_risk_v2/artifacts/d_pool.parquet",
        )

    @property
    def cohort(self) -> Path:
        """확정 등록/거절 결정만 남기고 기간·사건군을 붙인 라벨 표(v3 규칙, v6에서 확장)."""
        return self.workspace / "trademark_risk_v6/artifacts/cohort.parquet"

    @property
    def text_bundle(self) -> Path:
        """학습 기간에만 맞춘 TF-IDF 3종과 역할 사전(v3)."""
        return self.workspace / "trademark_risk_v3/artifacts/model.joblib"

    @property
    def text_svd(self) -> Path:
        """학습 기간에만 맞춘 TF-IDF 압축기(v6). 없으면 train 단계에서 새로 맞춘다."""
        return self.workspace / "trademark_risk_v6/artifacts/text_cache/svd.joblib"

    @property
    def kipris_merged_export(self) -> Path:
        """KIPRIS 내려받기 원본 39개를 합친 표. 각 행에 원본 파일명과 행 번호가 남아 있다."""
        return self.workspace / "01_상표권 등록/데이터/통합결과/상표_로우데이터_중복제거.csv"

    # ── 저장소에 포함된 자료 ──────────────────────────────────────────────────
    @property
    def role_lexicon(self) -> Path:
        """2017년 이전 지정상품에 실제로 쓰인 것이 확인된 역할 사전(산지·원재료·상품명 등)."""
        return REPO_ROOT / "resources/role_lexicon.json"

    @property
    def cmudict(self) -> Path:
        return REPO_ROOT / "resources/cmudict.dict"

    @property
    def official_records(self) -> Path:
        """KIPRIS 화면에서 직접 확인한 서지 원문(JSON). 출원인 정보가 있어 저장소에는 올리지 않는다."""
        return self.workspace / "trademark_risk_v9/data/official"

    @property
    def quarantine(self) -> Path:
        """이미 결과를 들여다본 사건 목록. 새 평가자료로 재사용하지 못하게 막는다."""
        return REPO_ROOT / "data/inspected_cases.json"

    # ── 출력 ────────────────────────────────────────────────────────────────
    @property
    def date_registry(self) -> Path:
        return self.output / "publication_dates.csv"

    @property
    def date_evidence(self) -> Path:
        return self.output / "publication_date_evidence.csv"

    @property
    def search_index(self) -> Path:
        return self.output / "search_index.joblib"

    @property
    def feature_chunks(self) -> Path:
        return self.output / "feature_chunks"

    @property
    def features(self) -> Path:
        return self.output / "features.parquet"

    @property
    def model(self) -> Path:
        return self.output / "model.joblib"

    @property
    def manifest(self) -> Path:
        return self.output / "frozen.json"

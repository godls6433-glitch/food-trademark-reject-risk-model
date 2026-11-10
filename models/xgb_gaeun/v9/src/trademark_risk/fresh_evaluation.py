"""새 평가자료의 입장 심사와 단 한 번의 채점.

지금까지의 평가 기간은 모두 개발 과정에서 본 자료다. 배포 여부는 **한 번도 보지 않은
사건**으로만 판단하기로 했고, 그 절차를 코드로 고정해 두었다.

1. 입장 심사: 이미 본 출원번호·이름·사건군, 미확정 결과, 입력 결측을 걸러 낸다.
2. 출처 확인서(provenance): 사람이 실제로 확인한 항목만 true로 적고 근거 파일을 붙인다.
3. 채점은 한 번만. 시작 기록이 남은 뒤에는 표본을 바꾸거나 모델을 고쳐 다시 잴 수 없다.

조건을 하나라도 채우지 못하면 실패로 끝난다(fail-closed). 현재까지 이 조건을 만족하는
자료는 확보하지 못했고, 따라서 '새 자료 성능'은 보고하지 않는다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import freeze, model
from .config import Paths
from .io_utils import read_json, sha256, write_json
from .textnorm import TARGET_CLASSES, normalize, parse_classes, parse_date

REQUIRED_COLUMNS = [
    "app", "name", "goods", "classes", "logo", "date", "status", "exam_status",
    "registration_number", "registration_date", "source_group", "source_document",
]  # fmt: skip
REQUIRED_ATTESTATIONS = [
    "independent_uninspected_source",
    "outcome_blinded_during_model_selection",
    "original_filing_inputs_verified",
    "historical_original_goods_and_availability_verified",
    "representative_sampling_verified",
]
MINIMUM_PER_CLASS = 100
ROC_AUC_TARGET = 0.7
EARLIEST_FRESH_DATE = "20240101"
_PLACEHOLDER_NAMES = ["", "상표명정보없음", "정보없음"]


def confirmed_label(row: pd.Series) -> int:
    """확정 결과만 라벨로 쓴다. 1 = 거절결정, 0 = 등록결정(+등록번호·등록일), -1 = 미확정."""
    refused = row["status"] == "거절" and "거절결정" in row["exam_status"]
    registered = (
        row["status"] == "등록"
        and "등록결정" in row["exam_status"]
        and bool(re.search(r"\d{6,}", row["registration_number"]))
        and bool(parse_date(row["registration_date"]))
    )
    return 1 if refused else 0 if registered else -1


def admit(cases: pd.DataFrame, known_apps: set, known_names: set, known_groups: set, quarantine: dict) -> pd.DataFrame:
    """행마다 제외 사유를 `exclusion` 열에 적는다. 빈 문자열이면 입장 가능."""
    missing = set(REQUIRED_COLUMNS) - set(cases.columns)
    if missing:
        raise ValueError("Missing columns: " + ",".join(sorted(missing)))

    cases = cases.copy().fillna("")
    cases["app"] = cases["app"].map(lambda value: re.sub(r"\D", "", value))
    cases["date"] = cases["date"].map(parse_date)
    cases["y"] = cases.apply(confirmed_label, axis=1)
    names = cases["name"].map(normalize)
    partial = cases["exam_status"].str.contains("부분|일부") | cases["status"].str.contains("부분|일부")

    checks = [
        ("invalid_application_number", ~cases["app"].str.fullmatch(r"\d{7}|\d{13}")),
        ("duplicate_app", cases["app"].duplicated(keep=False)),
        ("previously_exposed_identity", cases["app"].isin(known_apps) | names.isin(known_names) | cases["source_group"].isin(known_groups)),
        ("unconfirmed_or_partial", cases["y"].lt(0) | partial),
        (
            "missing_input",
            names.isin(_PLACEHOLDER_NAMES)
            | cases["goods"].str.strip().eq("")
            | ~cases["logo"].astype(str).isin(["0", "1"])
            | cases["source_group"].str.strip().eq("")
            | cases["source_document"].str.strip().eq(""),
        ),
        ("invalid_date_or_earlier_period", cases["date"].lt(EARLIEST_FRESH_DATE)),
        ("outside_classes", ~cases["classes"].map(lambda value: bool(parse_classes(value) & TARGET_CLASSES))),
    ]
    cases["exclusion"] = ""
    for reason, mask in checks:
        cases.loc[cases["exclusion"].eq("") & mask, "exclusion"] = reason

    # 결과와 이전 모델의 예측을 이미 들여다본 사건은 무엇으로 증명하든 새 자료가 될 수 없다.
    inspected = cases["app"].isin(quarantine["applications"]) | names.isin(quarantine["normalized_names"])
    cases.loc[inspected, "exclusion"] = "previously_inspected_development_pilot"
    return cases


def known_identities(paths: Paths) -> tuple[set, set, set]:
    """지금까지 풀·학습·평가에 한 번이라도 쓰인 출원번호, 정규화 이름, 사건군."""
    pool = pd.concat([pd.read_parquet(path, columns=["app", "name"]) for path in paths.pool_tables], ignore_index=True)
    cohort = pd.read_parquet(paths.cohort, columns=["app", "name", "source_group"])
    apps = set(pool["app"]) | set(cohort["app"])
    names = set(pool["name"].map(normalize)) | set(cohort["name"].map(normalize))
    return apps, names, apps | set(cohort["source_group"])


def check_provenance(provenance: dict, admitted: pd.DataFrame) -> list[str]:
    for key in REQUIRED_ATTESTATIONS:
        if provenance.get(key) is not True:
            raise ValueError("Unverified provenance: " + key)
    evidence = provenance.get("verification_evidence", [])
    if not evidence or not all(Path(item).is_file() for item in evidence):
        raise ValueError("Local source verification evidence missing")
    if not all(Path(item).is_file() for item in admitted["source_document"]):
        raise ValueError("Decision evidence missing")
    return evidence


def run(paths: Paths, cases_csv: str, provenance_json: str) -> dict:
    from .predict import Predictor

    freeze.verify(paths)
    destination = paths.output / "fresh_evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "started.json").exists():
        raise RuntimeError("Fresh evaluation consumed; never replace cohort or retune")

    cases = pd.read_csv(cases_csv, dtype=str, keep_default_na=False)
    review = admit(cases, *known_identities(paths), read_json(paths.quarantine))
    review.to_csv(destination / "admission.csv", index=False, encoding="utf-8-sig")
    admitted = review[review["exclusion"].eq("")].copy()
    if admitted["y"].nunique() != 2 or admitted["y"].value_counts().min() < MINIMUM_PER_CLASS:
        raise ValueError(f"At least {MINIMUM_PER_CLASS} independently verified cases in each class required")
    evidence = check_provenance(read_json(Path(provenance_json)), admitted)

    # 채점 전에 무엇을 무엇으로 재는지 못 박는다.
    write_json(
        destination / "started.json",
        dict(
            model_sha256=sha256(paths.model),
            cohort_sha256=sha256(cases_csv),
            provenance_sha256=sha256(provenance_json),
            evidence_sha256={str(item): sha256(item) for item in evidence},
            n=len(admitted),
        ),
    )
    predictor = Predictor(paths)
    admitted["v9"] = [predictor.predict(row, details=0)["rejection_probability"] for row in admitted.to_dict("records")]
    admitted.to_csv(destination / "predictions.csv", index=False, encoding="utf-8-sig")

    scores = model.score(admitted["y"], admitted["v9"])
    result = dict(metrics=scores, checks=dict(roc_above_target=scores["roc_auc"] > ROC_AUC_TARGET), site_changed=False)
    result["passed"] = all(result["checks"].values())
    write_json(destination / "result.json", result)
    return result

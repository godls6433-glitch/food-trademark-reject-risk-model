"""사후 정보 차단, 새 평가자료 입장 심사, 공고일 근거 조정."""

import json

import pandas as pd
import pytest

from trademark_risk.config import Paths
from trademark_risk.fresh_evaluation import admit
from trademark_risk.predict import clean_query
from trademark_risk.publication_dates import LIVE, XLSX, Identities, from_official_records, reconcile

QUERY = dict(app="q", name="테스트카페", date="20200102", goods="카페서비스업", classes="43", logo=0)
QUARANTINE = dict(applications=["4020240186129"], normalized_names=["likehur"])


def _case(**overrides):
    case = dict(
        app="4020260999999", name="새로운상표", goods="카페서비스업", classes="43", logo="0", date="20260101",
        status="거절", exam_status="거절결정", registration_number="", registration_date="",
        source_group="g1", source_document="decision.pdf",
    )  # fmt: skip
    return {**case, **overrides}


def _admit(cases, apps=(), names=(), groups=()):
    return admit(pd.DataFrame(cases), set(apps), set(names), set(groups), QUARANTINE)["exclusion"].tolist()


def test_outcome_fields_cannot_reach_the_model():
    noisy = dict(QUERY, status="거절", registration_number="40-1234567", reason="유사", cited_number="prior", y=1)
    assert clean_query(noisy) == clean_query(QUERY)


def test_query_without_goods_or_name_is_rejected():
    for broken in (dict(QUERY, goods=" "), dict(QUERY, name="정보없음"), dict(QUERY, logo=2)):
        with pytest.raises(ValueError):
            clean_query(broken)


def test_clean_unseen_case_is_admitted():
    assert _admit([_case()]) == [""]


def test_inspected_pilot_cases_never_become_fresh_data():
    assert _admit([_case(app="4020240186129")]) == ["previously_inspected_development_pilot"]
    assert _admit([_case(name=" LIKE HUR ")]) == ["previously_inspected_development_pilot"]  # 표기만 바꾼 경우


def test_previously_used_identities_are_excluded():
    assert _admit([_case()], apps=["4020260999999"]) == ["previously_exposed_identity"]
    assert _admit([_case()], names=["새로운상표"]) == ["previously_exposed_identity"]
    assert _admit([_case()], groups=["g1"]) == ["previously_exposed_identity"]


def test_only_confirmed_outcomes_count_as_labels():
    pending = _case(status="출원", exam_status="심사중")
    partial = _case(app="4020260999998", exam_status="일부거절결정")
    registered_without_number = _case(app="4020260999997", status="등록", exam_status="등록결정")
    assert _admit([pending, partial, registered_without_number]) == ["unconfirmed_or_partial"] * 3


def test_reconcile_rejects_conflicts_and_dates_before_filing():
    evidence = pd.DataFrame(
        [
            dict(app="a", publication="20210301", source_type=XLSX),
            dict(app="a", publication="20210301", source_type=LIVE),
            dict(app="b", publication="20210301", source_type=XLSX),
            dict(app="b", publication="20210401", source_type=LIVE),  # 근거끼리 날짜가 다름
            dict(app="c", publication="20190101", source_type=XLSX),  # 출원일보다 이른 공고일
        ]
    )
    filing = pd.Series({"a": "20200101", "b": "20200101", "c": "20200101"})
    accepted, conflicts = reconcile(evidence, filing)
    assert conflicts == {"b"}
    assert accepted[["app", "source_type"]].values.tolist() == [["a", LIVE]]


def test_official_record_uses_application_bulletin_never_registration_date(tmp_path):
    paths = Paths(workspace=tmp_path, output=tmp_path / "artifacts")
    paths.official_records.mkdir(parents=True)
    biblio = (
        "출원번호(일자)\t4020210070159 (2021.04.05)\n등록번호(일자)\t4019506170000 (2022.12.20)\n"
        "출원공고번호(일자)\t4020220091150 (2022.07.29)\n등록공고번호(일자)\t4020220132875 (2022.12.22)\n"
    )
    record = dict(app="4020210070159", observed_at="2026-10-12T07:07:45Z", url="https://www.kipris.or.kr", biblio=biblio)
    (paths.official_records / "4020210070159.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    identities = Identities(pd.DataFrame([dict(app="4020210070159", name="가상상표", date="20210405")]))
    (evidence,) = from_official_records(paths, {"4020210070159"}, identities)
    assert evidence["publication"] == "20220729" and evidence["field"] == "출원공고번호(일자)"
    assert evidence["identity_verified"] is True

    mismatched = Identities(pd.DataFrame([dict(app="4020210070159", name="가상상표", date="20210406")]))
    assert from_official_records(paths, {"4020210070159"}, mismatched)[0]["identity_verified"] is False

"""시점 규칙과 자기 사건 제외: 누수 방지의 핵심 계약."""

import pandas as pd
import pytest

from trademark_risk.pair_search import PAIR_COLUMNS, PairSearch
from trademark_risk.prior_search import PRIOR_COLUMNS, PriorSearch, link_families
from trademark_risk.search_index import assemble_pool

QUERY = dict(app="q", name="테스트카페", date="20200102", goods="카페서비스업", classes="43", logo=0)


def _pool(rows):
    base = dict(name="테스트카페", date="20190101", goods="카페서비스업", classes="43", codes="S120602")
    return pd.DataFrame([dict(base, app=app, publication=publication) for app, publication in rows])


@pytest.fixture
def search(analyzer):
    pool = _pool([("prior", "20200101"), ("same_day", "20200102"), ("future", "20200103"), ("unknown", ""), ("family", "20200101")])
    return PairSearch(PriorSearch(pool, families={"family": "q", "q": "q"}), analyzer)


def test_only_marks_published_strictly_before_filing_are_candidates(search):
    rows = search.candidates(QUERY)
    assert set(search.base.apps[rows]) == {"prior"}  # 당일·미래 공고, 같은 사건군은 제외


def test_missing_publication_date_is_never_indexed(search):
    assert "unknown" not in search.base.apps


def test_nothing_is_available_the_day_it_is_published(search):
    assert len(search.candidates(dict(QUERY, date="20200101"))) == 0


def test_invalid_filing_date_is_rejected_not_repaired(search):
    with pytest.raises(ValueError):
        search.candidates(dict(QUERY, date="20200231"))


def test_identical_prior_mark_scores_one(search):
    prior, listing, audit = search.base.extract(QUERY, details=3)
    pair, _, _ = search.extract(QUERY)
    assert list(prior) == PRIOR_COLUMNS and list(pair) == PAIR_COLUMNS
    assert prior["prior_exact_name"] == 1.0 and prior["prior_joint_combined_max"] == pytest.approx(1.0)
    assert pair["v7_pair_best"] == pytest.approx(1.0) and pair["v7_pair_exact_goods"] == 1.0
    assert audit["future_count"] == audit["own_family_count"] == 0
    assert [item["application"] for item in listing] == ["prior"]


def test_no_candidates_gives_explicit_empty_signal(search):
    prior, listing, _ = search.base.extract(dict(QUERY, name="완전히다른이름", date="20190101"))
    assert prior["prior_none"] == 1.0 and listing == []


def test_sound_alike_marks_are_retrieved_across_scripts(analyzer):
    pool = _pool([("p1", "20200101")]).assign(name="DAVID")
    search = PairSearch(PriorSearch(pool), analyzer)
    features, listing, _ = search.extract(dict(QUERY, name="데이빗"), details=1)
    assert listing[0]["application"] == "p1"
    assert features["v7_phone_max"] == pytest.approx(1.0) and features["v7_char_max"] == 0.0


def test_families_follow_explicit_original_application_links_only():
    families = link_families(pd.Series(["b", "c", "d"]), pd.Series(["40-2019-0000001", "", "4020190000001"]))
    assert families["b"] == families["d"] == families["4020190000001"]
    assert families["c"] == "c"


def test_assemble_pool_adds_only_registry_dates_and_drops_conflicts():
    pool = pd.concat(
        [
            _pool([("ok", "20200101"), ("no_date", ""), ("still_missing", "")]),
            _pool([("conflict", "20200101")]),
            _pool([("conflict", "20200101")]).assign(goods="식당업"),  # 같은 번호, 다른 내용
        ],
        ignore_index=True,
    )
    registry = pd.DataFrame([dict(app="no_date", publication="20191201")])
    combined, audit = assemble_pool(pool, registry)
    indexed = PriorSearch(combined)
    assert set(indexed.apps) == {"ok", "no_date"}
    assert audit["base_conflicting_applications"] == 1 and audit["repaired_applications"] == 1

    with pytest.raises(ValueError):  # 이미 기록된 공고일을 레지스트리로 덮어쓰지 않는다
        assemble_pool(pool, pd.DataFrame([dict(app="ok", publication="20191201")]))

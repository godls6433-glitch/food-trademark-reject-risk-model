"""선행상표 풀 → 검색 색인.

풀은 두 부분으로 이루어진다.

1. 기본 풀(v6): 원자료에 공고일이 기록된 출원.
2. 공고일 보완분(v9): 원자료에는 공고일이 비어 있었지만 KIPRIS 원본·공식 화면에서
   날짜를 확인한 출원. `publication_dates.py`가 만든 레지스트리에 있는 것만 넣는다.

같은 출원번호에 내용이 다른 행이 여러 개면 어느 쪽이 맞는지 임의로 고르지 않고
그 출원을 통째로 제외한다.
"""

from __future__ import annotations

import joblib
import pandas as pd

from .config import Paths
from .io_utils import sha256, write_json
from .pair_search import PairSearch
from .phonetic import NameAnalyzer
from .prior_search import INDEX_COLUMNS, PriorSearch, link_families, searchable_mask
from .textnorm import parse_date

_SOURCE_COLUMNS = {
    "app": "app",
    "name": "name",
    "date": "date",
    "출원공고일자": "publication",
    "지정상품": "goods",
    "classes": "classes",
    "유사군코드": "codes",
    "원출원번호": "original",
}


def load_pool(paths: Paths) -> pd.DataFrame:
    """색인에 필요한 열만 읽는다. 법적 상태·등록번호·출원인 열은 아예 불러오지 않는다."""
    tables = [pd.read_parquet(path, columns=list(_SOURCE_COLUMNS)) for path in paths.pool_tables]
    pool = pd.concat(tables, ignore_index=True).fillna("").rename(columns=_SOURCE_COLUMNS)
    for column in ("date", "publication"):
        parsed = {value: parse_date(value) for value in pool[column].unique()}
        pool[column] = pool[column].map(parsed)
    return pool


def _without_conflicts(table: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, set[str]]:
    """완전히 같은 행은 하나로 합치고, 그래도 여러 행이 남는 출원은 제외한다."""
    table = table.drop_duplicates(columns)
    conflicting = set(table.loc[table["app"].duplicated(keep=False), "app"])
    return table[~table["app"].isin(conflicting)], conflicting


def assemble_pool(pool: pd.DataFrame, registry: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """기본 풀에 공고일 보완분을 더한다."""
    base, base_conflicts = _without_conflicts(pool[INDEX_COLUMNS], INDEX_COLUMNS)

    identity = [column for column in INDEX_COLUMNS if column != "publication"]
    repaired, repair_conflicts = _without_conflicts(pool.loc[pool["app"].isin(registry["app"]), identity], identity)
    repaired = repaired.merge(registry[["app", "publication"]], on="app", validate="1:1")

    already_searchable = set(base.loc[searchable_mask(base), "app"]) & set(repaired["app"])
    if already_searchable:
        raise ValueError(f"Registry would overwrite recorded publication dates: {sorted(already_searchable)[:5]}")

    combined = pd.concat([base, repaired[INDEX_COLUMNS]], ignore_index=True)
    audit = dict(
        raw_rows=len(pool),
        base_conflicting_applications=len(base_conflicts),
        repaired_applications=len(repaired),
        repair_conflicting_applications=sorted(repair_conflicts),
        current_status_loaded=False,
        registration_date_imputed=False,
        original_goods_history_reconstructed=False,
    )
    return combined, audit


def build_in_memory(paths: Paths) -> tuple[PairSearch, dict]:
    pool = load_pool(paths)
    families = link_families(pool["app"], pool["original"])
    registry = pd.read_csv(paths.date_registry, dtype=str, keep_default_na=False)
    combined, audit = assemble_pool(pool, registry)
    del pool  # 63만 행 원본 표는 더 필요 없다. 색인을 만드는 동안의 메모리를 줄인다.

    base = PriorSearch(combined, families)
    search = PairSearch(base, NameAnalyzer.from_files(paths.cmudict, paths.role_lexicon))
    audit["indexed_rows"] = len(base.pool)
    return search, audit


def build(paths: Paths) -> PairSearch:
    """색인을 만들어 파일로 저장한다(약 1.1GB, 저장 중 메모리를 6GB 이상 쓴다)."""
    search, audit = build_in_memory(paths)
    paths.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(search, paths.search_index)
    audit["index_sha256"] = sha256(paths.search_index)
    write_json(paths.output / "index_audit.json", audit)
    return search


def load(paths: Paths) -> PairSearch:
    return joblib.load(paths.search_index)

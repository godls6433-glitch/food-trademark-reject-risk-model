"""라벨이 있는 모든 출원의 검색 특징(32열)을 계산한다.

25만여 건을 한 건씩 검색하므로 오래 걸린다. 500건 단위로 나눠 여러 프로세스에서
계산하고, 조각마다 저장해 두어 중단되면 남은 조각만 이어서 한다. 색인이나 라벨 표가
바뀌었는데 예전 조각을 섞어 쓰는 일이 없도록 해시가 다르면 이어 하기를 거부한다.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pandas as pd

from . import search_index
from .config import FEATURE_CHUNK_ROWS, QUERY_FIELDS, Paths
from .io_utils import read_json, sha256, write_json, write_parquet_atomic
from .pair_search import PairSearch

AUDIT_COLUMNS = ["future_count", "own_family_count"]

# 작업 프로세스가 공유하는 색인. fork로 프로세스를 만들면 1GB가 넘는 색인을
# 프로세스마다 복사하지 않고 그대로 읽을 수 있다.
_search: PairSearch | None = None


def search_features(search: PairSearch, query: dict) -> dict:
    """출원 한 건의 검색 특징 32개와 누수 점검 값."""
    prior, _, prior_audit = search.base.extract(query)
    pair, _, pair_audit = search.extract(query)
    audit = {column: prior_audit[column] + pair_audit[column] for column in AUDIT_COLUMNS}
    return {"app": query["app"], **prior, **pair, **audit}


def _compute_chunk(task: tuple[int, list[dict], str]) -> dict:
    start, queries, destination = task
    began = time.time()
    rows = [search_features(_search, query) for query in queries]
    write_parquet_atomic(pd.DataFrame(rows), Path(destination) / f"{start:07d}.parquet")
    return dict(start=start, rows=len(rows), seconds=round(time.time() - began, 1))


def build(paths: Paths, workers: int = 4) -> pd.DataFrame:
    global _search
    _search = search_index.load(paths)
    cohort = pd.read_parquet(paths.cohort)

    chunks = paths.feature_chunks
    chunks.mkdir(parents=True, exist_ok=True)
    signature = dict(index=sha256(paths.search_index), cohort=sha256(paths.cohort))
    signature_file = chunks / "signature.json"
    if signature_file.exists():
        if read_json(signature_file) != signature:
            raise RuntimeError("Index or cohort changed: clear feature_chunks/ before recomputing")
    else:
        write_json(signature_file, signature)

    fields = list(QUERY_FIELDS)
    tasks = [
        (start, cohort.iloc[start : start + FEATURE_CHUNK_ROWS][fields].to_dict("records"), str(chunks))
        for start in range(0, len(cohort), FEATURE_CHUNK_ROWS)
        if not (chunks / f"{start:07d}.parquet").exists()
    ]
    if workers == 1:
        for task in tasks:
            print(_compute_chunk(task), flush=True)
    else:
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for result in pool.imap_unordered(_compute_chunk, tasks):
                print(result, flush=True)

    features = pd.concat([pd.read_parquet(path) for path in sorted(chunks.glob("*.parquet"))], ignore_index=True)
    if features["app"].tolist() != cohort["app"].tolist():
        raise RuntimeError("Feature rows do not match the cohort")
    write_parquet_atomic(features, paths.features)
    return features

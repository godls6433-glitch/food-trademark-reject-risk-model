"""누락된 출원공고일을 근거가 있는 것만 보완한다 (v9의 변경점).

배경: 선행상표는 '조회 출원일 이전에 공개된 것'만 후보로 쓴다. 그런데 풀의 약 3분의 1은
공고일 칸이 비어 있어서, 실제로는 먼저 공개된 상표인데도 검색에서 통째로 빠져 있었다.
실제 거절 사건(LIKEHUR)의 인용상표가 바로 이 이유로 검색되지 않는 것을 확인하고 고쳤다.

원칙
- 날짜를 **추정하지 않는다.** 등록일이나 출원일로 공고일을 대신 채우지 않는다.
- 근거는 두 종류뿐: KIPRIS에서 내려받은 원본 엑셀의 해당 셀, KIPRIS 화면의 서지 원문.
- 근거마다 원본 파일 경로·행·SHA-256을 남겨 나중에 다시 대조할 수 있게 한다.
- 근거의 출원번호·상표명·출원일이 풀의 기록과 일치할 때만 받아들인다.
- 한 출원에 서로 다른 날짜가 나오면 어느 쪽도 쓰지 않는다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from . import freeze
from .config import Paths
from .io_utils import KIPRIS_HEADER_ROW, read_kipris_xlsx, sha256, write_json
from .textnorm import normalize, parse_date

APP, NAME, FILED, PUBLISHED = "출원(국제등록)번호", "상표명칭", "출원(국제등록)일자", "출원공고일자"
FIRST_DATA_ROW = KIPRIS_HEADER_ROW + 1

XLSX, LIVE = "KIPRIS_ORIGINAL_XLSX", "KIPRIS_LIVE_BIBLIOGRAPHY"
_NUMBER_AND_DATE = r"\s+(\d+)\s*\((\d{4}\.\d{2}\.\d{2})\)"
_BULLETIN_FIELDS = ("출원공고번호(일자)", "등록공고번호(일자)")  # 앞의 것을 우선한다


class Identities:
    """풀에 기록된 출원별 상표명·출원일. 근거가 같은 출원을 가리키는지 확인하는 데 쓴다."""

    def __init__(self, pool: pd.DataFrame):
        self._pool = pool.reset_index(drop=True)
        self._positions = self._pool.groupby("app").indices

    def _rows(self, app: str) -> pd.DataFrame:
        return self._pool.iloc[self._positions[app]]

    def first_name(self, app: str) -> str:
        return self._rows(app).iloc[0]["name"]

    def matches(self, app: str, filing_date: str, name: str | None = None) -> bool:
        rows = self._rows(app)
        same = rows["date"].map(parse_date).eq(parse_date(filing_date)).all()
        if name is not None:
            same = same and rows["name"].map(normalize).eq(normalize(name)).all()
        return bool(same)


def _xlsx_evidence(row: pd.Series, path: Path, root: Path, digest: str, sheet_row: int, observed_at: str, identities: Identities) -> dict:
    app = row[APP]
    return dict(
        app=app,
        publication=parse_date(row[PUBLISHED]),
        name=row[NAME],
        source_type=XLSX,
        source_path=str(path.relative_to(root)),
        source_sha256=digest,
        source_row=str(sheet_row),
        field=PUBLISHED,
        observed_at=observed_at,
        url="",
        identity_verified=identities.matches(app, row[FILED], row[NAME]),
    )


def _has_date(table: pd.DataFrame, targets: set[str]) -> pd.Series:
    return table[APP].isin(targets) & table[PUBLISHED].map(parse_date).ne("")


def from_merged_export(paths: Paths, targets: set[str], identities: Identities) -> list[dict]:
    """합친 표에서 찾은 날짜를, 그 행이 나온 원본 엑셀의 셀과 다시 대조한다."""
    merged = pd.read_csv(paths.kipris_merged_export, dtype=str, keep_default_na=False)
    found = merged[_has_date(merged, targets)]
    workbook_folder = paths.kipris_merged_export.parents[1]

    evidence = []
    for filename, rows in found.groupby("원본파일"):
        path = workbook_folder / filename
        workbook = read_kipris_xlsx(path)
        digest = sha256(path)
        for row in rows.to_dict("records"):
            original = workbook.iloc[int(row["원본행"]) - FIRST_DATA_ROW]
            if original[APP] != row[APP] or parse_date(original[PUBLISHED]) != parse_date(row[PUBLISHED]):
                raise ValueError(f"Merged export disagrees with {filename} row {row['원본행']}")
            observed = f"original_filename_{path.stem[:8]}"  # 파일명이 내려받은 시각
            evidence.append(_xlsx_evidence(original, path, paths.workspace, digest, int(row["원본행"]), observed, identities))
    return evidence


def from_extra_workbooks(paths: Paths, targets: set[str], identities: Identities) -> list[dict]:
    """작업 폴더 최상위에 따로 받아 둔 KIPRIS 엑셀들."""
    evidence = []
    for path in sorted(paths.workspace.glob("*.xlsx")):
        workbook = read_kipris_xlsx(path)
        if not {APP, PUBLISHED, NAME, FILED} <= set(workbook.columns):
            continue
        digest = sha256(path)
        for position, row in workbook[_has_date(workbook, targets)].iterrows():
            observed = "user_supplied_export_collection_time_unverified"
            evidence.append(_xlsx_evidence(row, path, paths.workspace, digest, position + FIRST_DATA_ROW, observed, identities))
    return evidence


def from_official_records(paths: Paths, targets: set[str], identities: Identities) -> list[dict]:
    """KIPRIS 화면에서 직접 확인해 저장한 서지 원문에서 공고일을 읽는다."""
    evidence = []
    for path in sorted(paths.official_records.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        app = record["app"]
        if app not in targets:
            continue
        bulletin = next(
            ((field, match) for field in _BULLETIN_FIELDS if (match := re.search(re.escape(field) + _NUMBER_AND_DATE, record["biblio"]))),
            None,
        )
        if bulletin is None:
            continue
        field, match = bulletin
        filing = re.search(re.escape("출원번호(일자)") + _NUMBER_AND_DATE, record["biblio"])
        if not filing or filing[1] != app:
            raise ValueError(f"Official record does not describe application {app}: {path.name}")
        evidence.append(
            dict(
                app=app,
                publication=parse_date(match[2]),
                name=identities.first_name(app),
                source_type=LIVE,
                source_path=str(path.relative_to(paths.workspace)),
                source_sha256=sha256(path),
                source_row="",
                field=field,
                observed_at=record["observed_at"],
                url=record["url"],
                identity_verified=identities.matches(app, filing[2], record.get("name") or None),
                publication_number=match[1],
            )
        )
    return evidence


def reconcile(evidence: pd.DataFrame, filing_dates: pd.Series) -> tuple[pd.DataFrame, set[str]]:
    """출원별로 날짜 하나를 확정한다. 날짜가 엇갈리는 출원과 출원일보다 이른 공고일은 버린다."""
    distinct = evidence.groupby("app")["publication"].nunique()
    conflicts = set(distinct[distinct > 1].index)
    accepted = evidence[~evidence["app"].isin(conflicts)]
    # 근거가 여럿이면 공식 화면 확인(LIVE)을 대표 근거로 남긴다.
    accepted = accepted.sort_values(["app", "source_type"], kind="stable").drop_duplicates("app")
    not_before_filing = accepted["publication"].to_numpy() >= filing_dates.loc[accepted["app"]].to_numpy()
    return accepted[not_before_filing], conflicts


def run(paths: Paths) -> dict:
    freeze.ensure_not_frozen(paths)
    columns = ["app", "name", "date", PUBLISHED]
    pool = pd.concat([pd.read_parquet(path, columns=columns) for path in paths.pool_tables], ignore_index=True).fillna("")
    missing = pool[pool[PUBLISHED].map(parse_date).eq("")]
    targets = set(missing["app"])
    identities = Identities(pool[pool["app"].isin(targets)])

    collected = [
        *from_merged_export(paths, targets, identities),
        *from_extra_workbooks(paths, targets, identities),
        *from_official_records(paths, targets, identities),
    ]
    rejected = [item for item in collected if not item["identity_verified"]]
    evidence = pd.DataFrame([item for item in collected if item["identity_verified"]])
    if evidence.empty:
        raise RuntimeError("No verified publication-date evidence found")

    filing_dates = pool.groupby("app")["date"].first().map(parse_date)
    accepted, conflicts = reconcile(evidence, filing_dates)

    paths.output.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(paths.date_registry, index=False, encoding="utf-8-sig")
    evidence.to_csv(paths.date_evidence, index=False, encoding="utf-8-sig")
    remaining = missing[~missing["app"].isin(accepted["app"])]
    audit = dict(
        pool_rows=len(pool),
        missing_rows=len(missing),
        missing_apps=len(targets),
        accepted_apps=len(accepted),
        evidence_records=len(evidence),
        conflicting_apps=sorted(conflicts),
        rejected_identity=rejected,
        remaining_apps=int(remaining["app"].nunique()),
        no_registration_date_imputation=True,
    )
    write_json(paths.output / "date_repair_audit.json", audit)
    return audit

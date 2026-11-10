"""파일 입출력 보조: 해시, JSON, 원자적 저장, KIPRIS 엑셀 읽기."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

_SHEET_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# KIPRIS 내려받기 파일은 8행이 머리글이고 9행부터 자료다.
KIPRIS_HEADER_ROW = 8


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    """임시 파일에 쓴 뒤 이름을 바꾼다. 중간에 끊겨도 반쯤 쓰인 파일이 남지 않는다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def read_kipris_xlsx(path: str | Path, header_row: int = KIPRIS_HEADER_ROW) -> pd.DataFrame:
    """KIPRIS 엑셀의 셀 값을 문자열 그대로 읽는다.

    스타일 정보가 깨진 파일이 있어 openpyxl 대신 XML을 직접 읽는다. 출원번호 같은
    식별번호가 숫자로 바뀌어 앞자리 0이 사라지는 일을 막기 위해 모든 값을 문자열로 둔다.
    계산 결과가 저장되지 않은 수식 셀은 조용히 비우지 않고 오류로 처리한다.
    """
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall("s:si", _SHEET_NS)]

        rows: dict[int, dict[str, str]] = {}
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//s:row", _SHEET_NS):
            cells = {}
            for cell in row:
                reference = cell.attrib["r"]
                column = "".join(c for c in reference if c.isalpha())
                value = cell.findtext("s:v", default="", namespaces=_SHEET_NS)
                if cell.find("s:f", _SHEET_NS) is not None and not value:
                    raise ValueError(f"Uncached formula: {reference}")
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(cell.find("s:is", _SHEET_NS).itertext())
                cells[column] = value
            rows[int(row.attrib["r"])] = cells

    headers = rows[header_row]
    data = [[cells.get(column, "") for column in headers] for number, cells in rows.items() if number > header_row and any(cells.values())]
    return pd.DataFrame(data, columns=list(headers.values()), dtype=str)

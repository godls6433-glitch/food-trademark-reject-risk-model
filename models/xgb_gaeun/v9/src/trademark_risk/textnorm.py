"""문자열 정규화와 파싱.

상표명·지정상품·상품류·날짜를 비교 가능한 형태로 바꾸는 가장 아래층 함수들이다.
학습과 예측이 반드시 같은 함수를 지나가도록 한 곳에 모아 두었다.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime

# 식음료 프로젝트의 분석 대상 상품류(니스 분류).
TARGET_CLASSES = frozenset({29, 30, 32, 43})

_GOODS_SEPARATOR = re.compile(r"[|;,\n]+")


def normalize(value: object) -> str:
    """NFC 정규화 → 소문자 → 글자·숫자만 남긴다. 예: ' LIKE-HUR ' → 'likehur'."""
    text = "" if value is None else str(value)
    return "".join(c for c in unicodedata.normalize("NFC", text).lower() if c.isalnum())


def to_jamo(text: str) -> str:
    """한글 음절을 초·중·종성으로 푼다(NFD). 발음이 비슷한 표기를 문자 수준에서 근사한다."""
    return unicodedata.normalize("NFD", text)


def goods_terms(value: object) -> frozenset[str]:
    """지정상품 목록을 구분자로 나눠 정규화한 상품명 집합으로 만든다."""
    text = "" if value is None else str(value)
    return frozenset(term for term in map(normalize, _GOODS_SEPARATOR.split(text)) if term)


def char_ngrams(text: str, n: int = 2) -> frozenset[str]:
    """문자 n-gram 집합. n보다 짧은 문자열은 문자열 자체를 하나의 조각으로 본다."""
    if len(text) < n:
        return frozenset([text]) if text else frozenset()
    return frozenset(text[i : i + n] for i in range(len(text) - n + 1))


def parse_classes(value: object) -> frozenset[int]:
    """'30|43' 같은 표기에서 1~45 범위의 상품류 번호만 뽑는다."""
    return frozenset(int(x) for x in re.findall(r"\d+", str(value)) if 1 <= int(x) <= 45)


def parse_date(value: object) -> str:
    """'2024.10.11' 등을 'YYYYMMDD'로. 달력에 없는 날짜나 형식 오류는 빈 문자열."""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 8:
        return ""
    try:
        datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return ""
    return digits

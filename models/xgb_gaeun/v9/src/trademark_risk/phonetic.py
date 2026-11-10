"""발음 근사와 요부(핵심 부분) 후보 (v7에서 도입).

'DAVID'와 '데이빗', 'O MI DANG'과 '오미당'처럼 표기는 다르지만 부르는 소리가 같은
상표를 찾기 위한 **검색용 근사**다. 심사 실무의 칭호 판단이나 심사관이 고른 요부를 대신하지 않는다.

- 영어 단어는 CMU 발음 사전, 한글은 음절별 로마자 변환으로 소리열을 만든다.
- 유성/무성(b·p, d·t, g·k), l·r, 비슷한 모음을 하나로 합쳐 일부러 느슨하게 맞춘다.
- 요부 후보는 눈에 보이는 경계(띄어쓰기·문자 종류)와, 사전에 있는 보통명칭을
  앞뒤에서 떼어 낸 나머지로만 만든다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from .textnorm import normalize

_INITIALS = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_VOWELS = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo",
    "we", "wi", "yu", "eu", "ui", "i",
]  # fmt: skip
_FINALS = [
    "", "k", "k", "ks", "n", "nj", "nh", "t", "l", "lk", "lm", "lp", "ls", "lt", "lp", "lh", "m",
    "p", "ps", "t", "t", "ng", "t", "t", "k", "t", "p", "h",
]  # fmt: skip

# CMU 사전의 음소 기호(ARPAbet) → 한글 로마자와 같은 표기 체계.
_ARPABET = dict(
    AA="a", AE="ae", AH="eo", AO="o", AW="au", AY="ai", B="b", CH="ch", D="d", DH="d", EH="e",
    ER="eo", EY="ei", F="p", G="g", HH="h", IH="i", IY="i", JH="j", K="k", L="r", M="m", N="n",
    NG="ng", OW="o", OY="oi", P="p", R="r", S="s", SH="s", T="t", TH="s", UH="u", UW="u", V="b",
    W="w", Y="y", Z="s", ZH="j",
)  # fmt: skip

_MERGE_CONSONANTS = str.maketrans({"b": "p", "v": "p", "f": "p", "d": "t", "g": "k", "z": "s", "l": "r", "j": "s"})

MAX_PARTS = 20


def load_cmudict(path: Path) -> dict[str, str]:
    """CMU 발음 사전을 {단어: 소리열}로 읽는다. 발음이 여럿이면 첫 번째를 쓴다."""
    pronunciations: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        word, *phones = line.split("#", 1)[0].split()
        word = re.sub(r"\(\d+\)$", "", word)
        sounds = "".join(_ARPABET.get(re.sub(r"\d", "", phone), "") for phone in phones)
        pronunciations.setdefault(word, sounds)
    return pronunciations


def load_generic_terms(lexicon_path: Path) -> list[str]:
    """요부에서 떼어 낼 보통명칭(상품명·서비스 내용). 긴 단어부터 맞춘다."""
    lexicon = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
    return sorted(set(lexicon["PRODUCT"] + lexicon["SERVICE_CONTENT"]), key=lambda term: (-len(term), term))


class NameAnalyzer:
    """이름 하나를 소리열과 요부 후보로 바꾼다. 같은 이름은 한 번만 계산한다."""

    def __init__(self, cmudict: dict[str, str], generic_terms: list[str]):
        self.cmudict = cmudict
        self.generic_terms = generic_terms
        self._generic = set(generic_terms)
        self._phonetic_cache: dict[str, str] = {}
        self._parts_cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_files(cls, cmudict_path: Path, lexicon_path: Path) -> NameAnalyzer:
        return cls(load_cmudict(cmudict_path), load_generic_terms(lexicon_path))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_phonetic_cache"], state["_parts_cache"] = {}, {}
        return state

    def phonetic(self, text: str) -> str:
        """느슨한 소리열. 예: 'DAVID'와 '데이빗'은 둘 다 'teipit'이 된다."""
        if text not in self._phonetic_cache:
            self._phonetic_cache[text] = self._phonetic(text)
        return self._phonetic_cache[text]

    def _phonetic(self, text: str) -> str:
        value = normalize(text)
        if value in self.cmudict:
            sounds = self.cmudict[value]
        else:
            sounds = "".join(_romanize_syllable(c) if "가" <= c <= "힣" else c for c in value)

        sounds = sounds.replace("ay", "ei").replace("ey", "ei")
        sounds = sounds.replace("ph", "f")
        sounds = re.sub("ch|sh", "s", sounds)
        sounds = sounds.replace("th", "t").replace("x", "ks")
        sounds = re.sub("c(?=[iey])", "s", sounds)
        sounds = sounds.replace("c", "k").replace("q", "k")
        sounds = sounds.translate(_MERGE_CONSONANTS)
        sounds = sounds.replace("ae", "e").replace("eo", "o").replace("eu", "u")
        return re.sub(r"(.)\1+", r"\1", sounds)  # 같은 소리의 반복은 하나로

    def parts(self, text: str) -> tuple[str, ...]:
        """요부 후보. 예: '찬미소 쭈꾸미' → ('찬미소쭈꾸미', '찬미소', '쭈꾸미'). 첫 원소는 전체 이름."""
        if text not in self._parts_cache:
            self._parts_cache[text] = self._parts(text)
        return self._parts_cache[text]

    def _parts(self, text: str) -> tuple[str, ...]:
        raw = str(text)
        full = normalize(raw)
        # 문자 종류가 바뀌는 지점에서 나눈 조각(2자 이상).
        segments = re.findall(r"[가-힣]+|[A-Za-z]+|[0-9]+", unicodedata.normalize("NFC", raw))
        # 'O MI DANG'처럼 띄어 쓴 영문은 붙인 형태도 후보로 둔다.
        joined = [normalize(phrase) for phrase in re.findall(r"[A-Za-z][A-Za-z ]*[A-Za-z]", raw)]
        candidates = list(dict.fromkeys([full] + [normalize(s) for s in segments if len(s) >= 2] + joined))

        # 보통명칭이 앞이나 뒤에 붙어 있으면 떼어 낸 나머지도 후보(2자 이상 남을 때만).
        for candidate in candidates.copy():
            for term in self.generic_terms:
                if candidate.endswith(term) and len(candidate) - len(term) >= 2:
                    candidates.append(candidate[: -len(term)])
                if candidate.startswith(term) and len(candidate) - len(term) >= 2:
                    candidates.append(candidate[len(term) :])

        unique = dict.fromkeys(candidates)
        return tuple(c for c in unique if c == full or c not in self._generic)[:MAX_PARTS]


def _romanize_syllable(syllable: str) -> str:
    index = ord(syllable) - 0xAC00
    return _INITIALS[index // 588] + _VOWELS[(index % 588) // 28] + _FINALS[index % 28]

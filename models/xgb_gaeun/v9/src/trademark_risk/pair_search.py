"""발음·요부까지 넓힌 선행상표 검색과 '이름×상품' 결합 특징 (v7에서 도입).

`PriorSearch`의 문자·자모 후보에, 요부 후보나 그 소리열이 **정확히 같은** 선행상표를
더한다. 그 뒤 후보마다 이름·발음·요부 유사도를 연속 점수로 다시 계산한다.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from rapidfuzz import fuzz, process

from .config import MAX_CANDIDATES
from .phonetic import NameAnalyzer
from .prior_search import PriorSearch, goods_parts
from .textnorm import normalize, parse_classes

PAIR_COLUMNS = [
    "v7_char_max", "v7_phone_max", "v7_keypart_max",
    "v7_pair_char", "v7_pair_phone", "v7_pair_keypart", "v7_pair_best",
    "v7_pair_top3", "v7_pair_strong_count", "v7_pair_exact_goods",
]  # fmt: skip

_PART, _SOUND = 0, 1


class PairSearch:
    """`PriorSearch` 위에 요부·발음 색인을 얹은 검색기."""

    def __init__(self, base: PriorSearch, analyzer: NameAnalyzer):
        self.base = base
        self.analyzer = analyzer

        parts: list[str] = []
        sounds: list[str] = []
        owners: list[int] = []  # 각 요부 후보가 속한 선행상표의 행 번호
        for row, name in enumerate(base.pool["name"]):
            for part in analyzer.parts(name):
                parts.append(part)
                sounds.append(analyzer.phonetic(part))
                owners.append(row)

        self.parts = parts
        self.sounds = sounds
        # 한 선행상표의 요부 후보들은 연속으로 놓이므로 [start, end) 구간으로 꺼낸다.
        per_row = np.bincount(np.array(owners, dtype=np.int32), minlength=len(base.pool))
        self.part_offsets = np.r_[0, np.cumsum(per_row)]

        exact: list[dict[str, list[int]]] = [defaultdict(list), defaultdict(list)]
        for kind, values in ((_PART, parts), (_SOUND, sounds)):
            for value, owner in zip(values, owners):
                if len(value) >= 2:
                    exact[kind][value].append(owner)
        self.exact = [{value: np.unique(rows).astype(np.int32) for value, rows in table.items()} for table in exact]

    def candidates(self, query: dict) -> np.ndarray:
        """기본 후보 + 요부/소리열이 정확히 일치하는 선행상표.

        추가 채널을 '정확 일치'로 제한한 이유: 소리열 2-gram으로 느슨하게 모으면
        흔한 조각 때문에 후보가 폭증한다. 느슨한 비교는 모인 후보 안에서만 한다.
        """
        base = self.base
        date = base.filing_date(query)
        found = list(base.candidates(query))
        for kind in (_PART, _SOUND):
            for part in self.analyzer.parts(query["name"]):
                key = self.analyzer.phonetic(part) if kind == _SOUND else part
                rows = self.exact[kind].get(key)
                if rows is None:
                    continue
                found.extend(base.eligible(base.before(rows, date), query)[:MAX_CANDIDATES])
        return np.unique(found).astype(np.int32)

    def extract(self, query: dict, details: int = 0) -> tuple[dict[str, float], list[dict], dict]:
        """조회 한 건 → (모델 특징 10개, 검수용 상위 후보 목록, 누수 점검 결과)."""
        base = self.base
        rows = self.candidates(query)
        count = len(rows)
        features = dict.fromkeys(PAIR_COLUMNS, 0.0)
        if not count:
            return features, [], {"candidate_count": 0, "future_count": 0, "own_family_count": 0}
        audit = base.audit(rows, query)

        # (1) 후보별 세 가지 이름 유사도: 전체 문자, 요부, 소리열.
        # 주의: 여기의 cdist는 rapidfuzz 기본 정밀도(float32)다. 동결 모델이 이 값으로
        # 학습됐으므로 정밀도를 바꾸면 특징값이 미세하게 달라진다.
        char = process.cdist([normalize(query["name"])], [base.names[row] for row in rows], scorer=fuzz.ratio)[0] / 100
        spans = [(self.part_offsets[row], self.part_offsets[row + 1]) for row in rows]
        part_ids = np.concatenate([np.arange(start, end) for start, end in spans])
        owner = np.repeat(np.arange(count), [end - start for start, end in spans])

        query_parts = self.analyzer.parts(query["name"])
        query_sounds = [s for s in map(self.analyzer.phonetic, query_parts) if len(s) >= 2]
        keypart = np.zeros(count)
        phone = np.zeros(count)
        scores = process.cdist(query_parts, [self.parts[i] for i in part_ids], scorer=fuzz.ratio).max(axis=0) / 100
        np.maximum.at(keypart, owner, scores)  # 후보의 요부들 중 가장 잘 맞는 것
        if query_sounds:
            candidate_sounds = [self.sounds[i] if len(self.sounds[i]) >= 2 else "" for i in part_ids]
            scores = process.cdist(query_sounds, candidate_sounds, scorer=fuzz.ratio).max(axis=0) / 100
            np.maximum.at(phone, owner, scores)

        # (2) 같은 후보의 지정상품 겹침과 곱한다. 서로 다른 후보의 최댓값을 섞지 않는다.
        overlap = base.goods_overlap(rows, str(query.get("goods", "")))
        goods = 0.6 * overlap.coverage + 0.4 * overlap.char_dice
        query_classes = parse_classes(query["classes"])
        same_class = np.array([bool(query_classes & base.class_sets[row]) for row in rows])

        # (3) 순위용 종합 점수. 이름 유사도를 세제곱해 '꽤 비슷함'과 '거의 같음'의 차이를
        # 벌리고, 상품류만 겹치는 경우는 약한 바닥값(0.25)만 준다.
        similarity = np.maximum.reduce([char, 0.95 * phone, 0.95 * keypart])
        pair = similarity**3 * (0.25 * same_class + 0.75 * goods)
        ranking = np.lexsort((rows, -char, -pair))
        top3 = ranking[:3]

        strong = (np.maximum.reduce([char, phone, keypart]) >= 0.85) & (overlap.coverage >= 0.5)
        exact = char >= 1 - 1e-6
        values = [
            char.max(),
            phone.max(),
            keypart.max(),
            (char * goods).max(),
            (phone * goods).max(),
            (keypart * goods).max(),
            pair.max(),
            pair[top3].mean(),
            np.log1p(strong.sum()) / 10,
            overlap.coverage[exact].max() if exact.any() else 0.0,
        ]
        features.update(zip(PAIR_COLUMNS, map(float, values)))

        listing = []
        for rank, position in enumerate(ranking[:details], 1):
            record = base.pool.iloc[int(rows[position])]
            listing.append(
                dict(
                    rank=rank,
                    application=record["app"],
                    name=record["name"],
                    filing_date=record["date"],
                    publication_date=record["publication"],
                    char_similarity=float(char[position]),
                    phonetic_similarity=float(phone[position]),
                    keypart_similarity=float(keypart[position]),
                    goods_query_coverage=float(overlap.coverage[position]),
                    goods_character_dice=float(overlap.char_dice[position]),
                    pair_score=float(pair[position]),
                    exact_overlapping_goods=sorted(overlap.query_terms & goods_parts(record["goods"])[0]),
                    historical_status="UNKNOWN",
                    original_goods_verified=False,
                    interpretation="검색용 후보 점수. 발음·요부 및 법적 충돌 여부는 미확인",
                )
            )
        return features, listing, audit

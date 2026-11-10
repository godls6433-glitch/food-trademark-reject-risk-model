"""출원일 이전에 공개된 선행상표 검색과 이름·지정상품 유사도 특징 (v6에서 도입).

핵심 규칙
- 후보는 **출원일과 공고일이 모두 조회 출원일보다 이른** 기록만. 당일도 제외한다.
- 조회 대상 자신과 원출원으로 연결된 사건군은 후보에서 뺀다.
- 색인에는 번호·이름·날짜·지정상품·상품류만 올린다. 현재 법적 상태, 등록 여부,
  출원인은 읽지 않으므로 사후 결과가 특징으로 새어 들어올 수 없다.
- 이름 유사도와 상품 겹침은 **같은 선행상표 한 건 안에서** 결합한다.
"""

from __future__ import annotations

import re
from array import array
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy import sparse

from .config import MAX_CANDIDATES
from .textnorm import TARGET_CLASSES, char_ngrams, goods_terms, normalize, parse_classes, parse_date, to_jamo

INDEX_COLUMNS = ["app", "name", "date", "publication", "goods", "classes", "codes"]

NAME_COLUMNS = [
    "prior_char_max", "prior_jamo_max", "prior_partial_max", "prior_exact_name",
    "prior_char_top5_mean", "prior_name_ge80_logcount", "prior_candidate_logcount", "prior_none",
]  # fmt: skip
GOODS_COLUMNS = [
    "prior_joint_coverage_max", "prior_joint_goodsdice_max", "prior_joint_jaccard_max",
    "prior_joint_combined_max", "prior_joint_top5_mean", "prior_bestchar_goods_coverage",
    "prior_bestchar_goods_dice", "prior_bestjamo_goods_coverage", "prior_name80_goods50_logcount",
    "prior_exact_name_goods_coverage", "prior_name60_goods_coverage_max", "prior_sameclass_char_max",
    "prior_bestjoint_class_overlap", "prior_goods_missing_fraction",
]  # fmt: skip
PRIOR_COLUMNS = NAME_COLUMNS + GOODS_COLUMNS

# 검색 채널: (이름 표현, n-gram 길이). 0 = 문자 2-gram, 1 = 자모 3-gram.
_CHANNELS = ((lambda name: name, 2), (to_jamo, 3))


class LeakageError(RuntimeError):
    """미래 기록이나 자기 사건이 후보에 섞였을 때. 조용히 넘어가지 않고 멈춘다."""


@dataclass
class GoodsOverlap:
    """조회 출원과 후보 각각의 지정상품 겹침(후보 수 길이의 배열)."""

    query_terms: frozenset[str]
    coverage: np.ndarray  # |Q∩C| / |Q| — 내 상품 중 후보에도 있는 비율
    jaccard: np.ndarray  # |Q∩C| / |Q∪C|
    char_dice: np.ndarray  # 상품명 문자 2-gram 집합의 Dice 계수
    missing: np.ndarray  # 후보의 지정상품이 비어 있는지


@lru_cache(maxsize=4096)
def goods_parts(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """지정상품 문자열 → (상품명 집합, 상품명들의 문자 2-gram 합집합)."""
    terms = goods_terms(text)
    bigrams = frozenset().union(*(char_ngrams(term, 2) for term in terms)) if terms else frozenset()
    return terms, bigrams


def searchable_mask(pool: pd.DataFrame) -> pd.Series:
    """색인에 올릴 수 있는 행: 유효한 출원일·공고일, 공고일 ≥ 출원일, 이름 있음, 대상 상품류."""
    names = pool["name"].map(normalize)
    mask = pool["date"].str.match(r"^\d{8}$") & pool["publication"].str.match(r"^\d{8}$")
    mask &= pool["publication"].ge(pool["date"]) & names.ne("") & names.ne("정보없음")
    mask &= pool["classes"].map(lambda value: bool(parse_classes(value) & TARGET_CLASSES))
    return mask


def link_families(applications: pd.Series, original_applications: pd.Series) -> dict[str, str]:
    """원출원번호로 이어진 출원들을 한 사건군으로 묶는다(union-find).

    분할·변경출원처럼 명시적으로 연결된 번호만 묶는다. 이름이 같다는 이유만으로
    같은 권리자라고 보지는 않는다. 사건군 대표는 구성원 중 가장 작은 번호 문자열이다.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for application, original in zip(applications, original_applications):
        find(application)
        for number in re.findall(r"\d[\d-]{4,}\d", original):
            union(application, re.sub(r"\D", "", number))
    return {node: find(node) for node in list(parent)}


class PriorSearch:
    """선행상표 풀의 역색인. 한 번 만들고 여러 조회에 쓴다."""

    def __init__(self, pool: pd.DataFrame, families: dict[str, str] | None = None, max_candidates: int = MAX_CANDIDATES):
        self.max_candidates = max_candidates
        self.families = families or {}

        pool = pool[INDEX_COLUMNS].fillna("")
        pool = pool[searchable_mask(pool)].copy()
        # 공고일 순으로 정렬해 두면 '조회일 이전' 구간을 이분 탐색으로 잘라낼 수 있다.
        pool = pool.sort_values(["publication", "app", "name"], kind="stable").reset_index(drop=True)
        self.pool = pool

        self.names = pool["name"].map(normalize).tolist()
        self.jamos = [to_jamo(name) for name in self.names]
        self.filed = pool["date"].astype(int).to_numpy()
        self.published = pool["publication"].astype(int).to_numpy()
        self.apps = pool["app"].to_numpy()
        self.family = np.array([self.families.get(app, app) for app in pool["app"]])
        self.class_sets = [parse_classes(value) for value in pool["classes"]]

        self._rows_by_family: dict[str, list[int]] = defaultdict(list)
        for row, family in enumerate(self.family):
            self._rows_by_family[family].append(row)

        self._index_goods(pool["goods"])
        self._index_names()

    # ── 색인 구축 ─────────────────────────────────────────────────────────────
    def _index_goods(self, goods_column: pd.Series) -> None:
        """지정상품을 (행 × 상품명), (행 × 문자 2-gram) 희소 행렬 두 개로 저장한다.

        문자열 집합끼리 수억 번 교집합을 구하는 대신 정수 행렬의 열 선택·합으로
        같은 값을 얻는다. 0번 = 상품명 일치, 1번 = 문자 2-gram.
        """
        self.goods_vocab: list[dict[str, int]] = [{}, {}]
        indices = [array("i"), array("i")]
        indptr = [array("i", [0]), array("i", [0])]
        for text in goods_column:
            for kind, items in enumerate(goods_parts(text)):
                vocab = self.goods_vocab[kind]
                indices[kind].extend(sorted(vocab.setdefault(item, len(vocab)) for item in items))
                indptr[kind].append(len(indices[kind]))

        self.goods_matrices = []
        for kind in (0, 1):
            columns = np.frombuffer(indices[kind], dtype=np.int32).copy()
            pointers = np.frombuffer(indptr[kind], dtype=np.int32).copy()
            data = np.ones(len(columns), dtype=np.int32)
            shape = (len(goods_column), len(self.goods_vocab[kind]))
            self.goods_matrices.append(sparse.csr_matrix((data, columns, pointers), shape=shape))
        self.goods_sizes = [np.diff(matrix.indptr) for matrix in self.goods_matrices]

    def _index_names(self) -> None:
        """이름의 문자 2-gram, 자모 3-gram → 그 조각을 가진 행 번호 목록(역색인)."""
        self.postings: list[dict[str, np.ndarray]] = []
        self.gram_counts: list[np.ndarray] = []
        for texts, (_, n) in zip((self.names, self.jamos), _CHANNELS):
            posting: dict[str, list[int]] = defaultdict(list)
            counts = []
            for row, text in enumerate(texts):
                grams = char_ngrams(text, n)
                counts.append(len(grams))
                for gram in grams:
                    posting[gram].append(row)
            self.postings.append({gram: np.array(rows, dtype=np.int32) for gram, rows in posting.items()})
            self.gram_counts.append(np.array(counts, dtype=np.float32))

    # ── 조회 ─────────────────────────────────────────────────────────────────
    @staticmethod
    def filing_date(query: dict) -> int:
        """조회 입력 검증. 날짜를 추정하거나 고쳐 쓰지 않고, 잘못되면 거부한다."""
        date = parse_date(query.get("date", ""))
        if not date or not normalize(query.get("name", "")):
            raise ValueError("Valid filing date and nonempty name are required")
        if not parse_classes(query.get("classes", "")) & TARGET_CLASSES:
            raise ValueError("At least one target class (29/30/32/43) is required")
        return int(date)

    def before(self, rows: np.ndarray, date: int) -> np.ndarray:
        """공고일 순으로 정렬된 행 목록에서 조회일 이전에 공고된 앞부분만 남긴다."""
        return rows[: np.searchsorted(self.published[rows], date, side="left")]

    def eligible(self, rows: np.ndarray, query: dict) -> np.ndarray:
        """시점 조건과 자기 사건군 제외를 적용한다."""
        date = self.filing_date(query)
        keep = (self.filed[rows] < date) & (self.published[rows] < date)
        app = str(query.get("app", ""))
        if app:
            own = self._rows_by_family.get(self.families.get(app, app), [])
            if own:
                keep &= ~np.isin(rows, own)
        return rows[keep]

    def candidates(self, query: dict) -> np.ndarray:
        """문자·자모 채널에서 조각 겹침(Dice)이 큰 후보를 채널별 최대 512건씩 모은다."""
        date = self.filing_date(query)
        name = normalize(query["name"])
        found: list[int] = []
        for channel, (represent, n) in enumerate(_CHANNELS):
            grams = char_ngrams(represent(name), n)
            hits = [self.before(self.postings[channel][g], date) for g in grams if g in self.postings[channel]]
            hits = [rows for rows in hits if len(rows)]
            if not hits:
                continue
            shared = np.bincount(np.concatenate(hits))
            rows = self.eligible(np.flatnonzero(shared), query)
            score = 2 * shared[rows] / (len(grams) + self.gram_counts[channel][rows])
            if len(rows) > self.max_candidates:
                cutoff = np.partition(score, -self.max_candidates)[-self.max_candidates]
                rows, score = rows[score >= cutoff], score[score >= cutoff]
            found.extend(rows[np.lexsort((rows, -score))[: self.max_candidates]])
        return np.unique(found).astype(np.int32)

    def audit(self, rows: np.ndarray, query: dict) -> dict[str, int]:
        """후보에 미래 기록·자기 사건군이 없는지 다시 센다. 하나라도 있으면 중단한다."""
        date = self.filing_date(query)
        app = str(query.get("app", ""))
        future = int(((self.filed[rows] >= date) | (self.published[rows] >= date)).sum())
        own = int((self.family[rows] == self.families.get(app, app)).sum()) if app else 0
        if future or own:
            raise LeakageError(f"future={future}, own_family={own} for application {app!r}")
        return {"candidate_count": len(rows), "future_count": future, "own_family_count": own}

    def goods_overlap(self, rows: np.ndarray, goods_text: str) -> GoodsOverlap:
        terms, bigrams = goods_parts(goods_text)
        shared = []
        for kind, items in enumerate((terms, bigrams)):
            columns = [self.goods_vocab[kind][item] for item in items if item in self.goods_vocab[kind]]
            shared.append(np.asarray(self.goods_matrices[kind][rows][:, columns].sum(axis=1)).ravel())
        shared_terms, shared_bigrams = shared
        term_sizes, bigram_sizes = (sizes[rows] for sizes in self.goods_sizes)

        zeros = np.zeros(len(rows))
        union = len(terms) + term_sizes - shared_terms
        both = len(bigrams) + bigram_sizes
        return GoodsOverlap(
            query_terms=terms,
            coverage=shared_terms / len(terms) if terms else zeros,
            jaccard=np.divide(shared_terms, union, out=zeros.copy(), where=union > 0),
            char_dice=np.divide(2 * shared_bigrams, both, out=zeros.copy(), where=both > 0),
            missing=term_sizes == 0,
        )

    def extract(self, query: dict, details: int = 0) -> tuple[dict[str, float], list[dict], dict]:
        """조회 한 건 → (모델 특징 22개, 검수용 상위 후보 목록, 누수 점검 결과)."""
        rows = self.candidates(query)
        count = len(rows)
        features = dict.fromkeys(PRIOR_COLUMNS, 0.0)
        features["prior_none"] = float(not count)
        features["prior_candidate_logcount"] = float(np.log1p(count) / 10)
        if not count:
            return features, [], {"candidate_count": 0, "future_count": 0, "own_family_count": 0}
        audit = self.audit(rows, query)

        name = normalize(query["name"])
        candidate_names = [self.names[row] for row in rows]
        char = process.cdist([name], candidate_names, scorer=fuzz.ratio, dtype=np.float64)[0] / 100
        jamo = process.cdist([to_jamo(name)], [self.jamos[row] for row in rows], scorer=fuzz.ratio, dtype=np.float64)[0] / 100
        partial = process.cdist([name], candidate_names, scorer=fuzz.partial_ratio, dtype=np.float64)[0] / 100
        goods = self.goods_overlap(rows, str(query.get("goods", "")))

        similarity = 0.65 * char + 0.35 * jamo
        joint = similarity * (0.5 * goods.coverage + 0.5 * goods.char_dice)
        query_classes = parse_classes(query["classes"])
        same_class = np.array([bool(query_classes & self.class_sets[row]) for row in rows])
        exact = char >= 1 - 1e-12

        ranking = np.lexsort((rows, -joint))
        top5 = ranking[:5]
        best_char, best_jamo, best_joint = int(np.argmax(char)), int(np.argmax(jamo)), int(top5[0])
        features.update(
            prior_char_max=float(char.max()),
            prior_jamo_max=float(jamo.max()),
            prior_partial_max=float(partial.max()),
            prior_exact_name=float(exact.any()),
            prior_char_top5_mean=float(np.sort(char)[-5:].mean()),
            prior_name_ge80_logcount=float(np.log1p((similarity >= 0.8).sum()) / 10),
            prior_joint_coverage_max=float((similarity * goods.coverage).max()),
            prior_joint_goodsdice_max=float((similarity * goods.char_dice).max()),
            prior_joint_jaccard_max=float((similarity * goods.jaccard).max()),
            prior_joint_combined_max=float(joint.max()),
            prior_joint_top5_mean=float(joint[top5].mean()),
            prior_bestchar_goods_coverage=float(goods.coverage[best_char]),
            prior_bestchar_goods_dice=float(goods.char_dice[best_char]),
            prior_bestjamo_goods_coverage=float(goods.coverage[best_jamo]),
            prior_name80_goods50_logcount=float(np.log1p(((similarity >= 0.8) & (goods.coverage >= 0.5)).sum()) / 10),
            prior_exact_name_goods_coverage=float(goods.coverage[exact].max()) if exact.any() else 0.0,
            prior_name60_goods_coverage_max=float(goods.coverage[similarity >= 0.6].max()) if (similarity >= 0.6).any() else 0.0,
            prior_sameclass_char_max=float(char[same_class].max()) if same_class.any() else 0.0,
            prior_bestjoint_class_overlap=float(same_class[best_joint]),
            prior_goods_missing_fraction=float(np.mean(goods.missing)),
        )

        listing = []
        for position in ranking[:details]:
            record = self.pool.iloc[int(rows[position])]
            listing.append(
                dict(
                    application=record["app"],
                    name=record["name"],
                    filing_date=record["date"],
                    publication_date=record["publication"],
                    classes=record["classes"],
                    char_similarity=float(char[position]),
                    jamo_similarity=float(jamo[position]),
                    goods_query_coverage=float(goods.coverage[position]),
                    goods_jaccard=float(goods.jaccard[position]),
                    goods_character_dice=float(goods.char_dice[position]),
                    joint_score=float(joint[position]),
                    exact_overlapping_goods=sorted(goods.query_terms & goods_parts(record["goods"])[0]),
                    historical_status="UNKNOWN",
                    original_goods_verified=False,
                )
            )
        return features, listing, audit

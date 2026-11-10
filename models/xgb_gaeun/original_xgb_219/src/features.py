"""Independent, label-free, strictly earlier-filing feature construction."""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date

import numpy as np
from rapidfuzz import fuzz

CLASSES = (29, 30, 32, 43)
# Frozen before evaluating any outer fold. Manually specified, no external TM data.
WORDS = {
    "goods": "김치 고기 한우 소고기 돼지 삼겹 갈비 족발 보쌈 치킨 닭 곱창 막창 국수 냉면 우동 라면 짬뽕 짜장 만두 떡 빵 과자 쿠키 케이크 피자 버거 샌드위치 김밥 초밥 스시 회 해물 생선 참치 연어 쭈꾸미 주꾸미 낙지 문어 오징어 새우 게 전복 조개 굴 순대 국밥 감자탕 설렁탕 곰탕 죽 찌개 샤브 두부 콩 쌀 밀 밥 반찬 도시락 샐러드 소스 장아찌 젓갈 홍삼 인삼 건강 유산균 비타민 우유 치즈 버터 요거트 아이스크림 커피 카페 차 녹차 홍차 음료 주스 쥬스 맥주 소주 와인 막걸리 물 생수 식품 푸드 coffee cafe bakery bread chicken pizza burger food tea beer milk fresh juice kitchen restaurant hotel".split(),
    "praise": "맛 맛있는 맛집 맛나 최고 최상 일등 명품 명가 명인 명장 원조 본가 전통 정통 정성 진짜 진한 착한 좋은 행복 건강 신선 자연 천연 순수 유기농 프리미엄 수제 엄마 어머니 할머니 장인 달인 고급 특선 특별 최고급 리얼 best good premium original natural organic delicious happy".split(),
    "place": "서울 부산 대구 인천 광주 대전 울산 세종 제주 강릉 속초 춘천 원주 전주 남원 안동 경주 포항 통영 여수 순천 목포 광양 담양 나주 나고야 도쿄 오사카 삿포로 일본 중국 홍콩 상해 베트남 태국 이탈리아 프랑스 미국 뉴욕 파리 로마 한강 강남 강북 종로 명동 홍대 연남 성수 압구정 청담 이태원 수원 인제 홍천 횡성 평창 강원 경기 충청 전라 경상 한국 코리아 korea seoul busan jeju newyork paris".split(),
    "business": "전문 전문점 식당 음식점 식품 식품점 상회 상점 가게 마켓 시장 유통 산업 회사 주식회사 가든 하우스 카페 커피숍 베이커리 제과 제빵 방앗간 포차 주점 호프 펍 레스토랑 호텔 리조트 펜션 cafe shop market store house bar pub company".split(),
}
TOKEN_FEATURES = tuple(dict.fromkeys(WORDS["goods"] + WORDS["praise"][:35] + WORDS["business"][:25]))


def normalize(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKC", s).lower() if c.isalnum())


def aliases(s: str) -> tuple[str, ...]:
    s = unicodedata.normalize("NFKC", s).lower()
    out = [normalize(s), "".join(re.findall("[가-힣]+", s)), "".join(re.findall("[a-z]+", s))]
    return tuple(dict.fromkeys(x for x in out if x))


def class_values(s: str) -> tuple[int, ...]:
    return tuple(sorted(set(int(x) for x in re.findall(r"\d+", s) if int(x) in CLASSES)))


def grams(s: str) -> tuple[str, ...]:
    # Prefix/suffix features include one-character names, without learned vocabulary.
    result = {"b:" + s[:2], "e:" + s[-2:]} if s else set()
    for n in (2, 3):
        result.update(f"{n}:" + s[i:i+n] for i in range(len(s)-n+1))
    return tuple(sorted(result))


def static_features(name: str, classes: tuple[int, ...], logo: float) -> dict[str, float]:
    raw = unicodedata.normalize("NFKC", name).lower()
    s = normalize(name)
    n = len(s)
    ko = sum("가" <= c <= "힣" for c in s)
    en = sum("a" <= c <= "z" for c in s)
    digits = sum(c.isdigit() for c in s)
    f = {
        "name_length": n, "raw_length": len(raw), "name_missing": int(not s),
        "korean_length": ko, "latin_length": en, "digit_length": digits,
        "other_length": n-ko-en-digits, "korean_ratio": ko/max(n, 1),
        "latin_ratio": en/max(n, 1), "digit_ratio": digits/max(n, 1),
        "unique_char_ratio": len(set(s))/max(n, 1), "token_count": len(raw.split()),
        "has_parenthesis": int("(" in raw or ")" in raw),
        "has_mixed_script": int(ko > 0 and en > 0), "has_logo": logo,
        "class_count": len(classes), "short_name": int(n <= 2),
        "punctuation_count": sum(not c.isalnum() and not c.isspace() for c in raw),
    }
    for c in CLASSES:
        f[f"class_{c}"] = int(c in classes)
    for category, words in WORDS.items():
        found = [w for w in words if w in s]
        cover = np.zeros(n, dtype=bool)
        for w in found:
            start = s.find(w)
            while start != -1:
                cover[start:start+len(w)] = True
                start = s.find(w, start+1)
        f[f"{category}_word_count"] = len(found)
        f[f"{category}_coverage"] = float(cover.mean()) if n else 0
        f[f"{category}_whole_name"] = int(s in words and bool(s))
    for i, w in enumerate(TOKEN_FEATURES):
        f[f"word_{i:03d}"] = int(w in s)
    return f


class HistoricalIndex:
    """Only insert records before a query's date. No labels or legal statuses here."""

    def __init__(self):
        self.postings = defaultdict(list)
        self.exact = defaultdict(list)
        self.class_count = Counter()
        self.rows = []
        self.max_day = -1000000

    def add(self, row):
        idx = len(self.rows)
        self.rows.append(row)
        self.max_day = max(self.max_day, row["day"])
        aa = row["aliases"]
        gg = set(g for a in aa for g in grams(a))
        for c in row["classes"]:
            self.class_count[c] += 1
            for a in aa:
                self.exact[c, a].append(idx)
            for g in gg:
                self.postings[c, g].append(idx)

    def query(self, row):
        assert self.max_day < row["day"], "Index includes same-day/future records"
        aa, cc = row["aliases"], row["classes"]
        gg = sorted(set(g for a in aa for g in grams(a)))
        candidates = Counter()
        exact_ids = set()
        history_count = sum(self.class_count[c] for c in cc)
        counts = []
        for c in cc:
            for a in aa:
                ids = self.exact.get((c, a), ())
                exact_ids.update(ids)
            entries = [(len(self.postings.get((c, g), ())), g) for g in gg]
            counts.extend(count for count, _ in entries)
            # Deterministic approximate retrieval; no future corpus statistics.
            entries = sorted((count, g) for count, g in entries if count)[:8]
            for count, g in entries:
                ids = self.postings[c, g]
                # Preserve early and recent representatives for common fragments.
                selected = ids if len(ids) <= 1200 else ids[:300] + ids[-900:]
                weight = 1 / math.log2(count + 2)
                for idx in selected:
                    candidates[idx] += weight
        selected = [idx for idx, _ in candidates.most_common(240)]
        selected = sorted(set(selected) | set(sorted(exact_ids)[-300:]))
        ranked = []
        app = row["application"]
        for idx in selected:
            p = self.rows[idx]
            if p["application"] == app and app:
                continue
            score = max((fuzz.ratio(a, b)/100 for a in aa for b in p["aliases"]), default=0)
            ranked.append((score, idx))
        ranked.sort(key=lambda x: (-x[0], self.rows[x[1]]["application"], x[1]))
        top, seen = [], set()
        for score, idx in ranked:
            p = self.rows[idx]
            if p["application"] in seen:
                continue
            seen.add(p["application"])
            assert p["day"] < row["day"], "Future/same-day prior leaked"
            top.append((score, p))
            if len(top) == 5:
                break
        exact_applications = {self.rows[i]["application"] for i in exact_ids if self.rows[i]["application"] != app}
        scores = [x[0] for x in top]
        f = {"history_log_count": math.log1p(history_count),
             "exact_prior_log_count": math.log1p(len(exact_applications)),
             "retrieved_log_count": math.log1p(len(candidates)),
             "prior_top5_mean": float(np.mean(scores)) if scores else 0,
             "prior_top5_count": len(scores),
             "prior_gap_1_2": scores[0]-scores[1] if len(scores) > 1 else 0,
             "gram_seen_fraction": float(np.mean(np.array(counts) > 0)) if counts else 0,
             "gram_log_frequency_mean": float(np.log1p(counts).mean()) if counts else 0,
             "gram_log_frequency_min": math.log1p(min(counts)) if counts else 0,
             "gram_log_frequency_max": math.log1p(max(counts)) if counts else 0,
             "prior_age_log_days": math.log1p(row["day"]-top[0][1]["day"]) if top else 0}
        for i in range(5):
            f[f"prior_similarity_{i+1}"] = scores[i] if i < len(scores) else 0
        for threshold in (50, 70, 80, 90, 100):
            f[f"prior_top5_ge_{threshold}"] = sum(x >= threshold/100 for x in scores)
        return f, top


def record(name, classes, application, day, logo, key):
    return {"name": name, "aliases": aliases(name), "classes": class_values(classes),
            "application": application, "day": int(day), "logo": float(logo), "key": key}

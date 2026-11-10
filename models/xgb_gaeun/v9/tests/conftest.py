"""테스트용 가상 자료. 실제 상표·출원번호가 아니며 어떤 보고 수치에도 들어가지 않는다."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from trademark_risk.config import Paths
from trademark_risk.phonetic import NameAnalyzer
from trademark_risk.text_features import structure_features
from trademark_risk.textnorm import goods_terms, normalize, to_jamo

SYLLABLES = list("가나다라마바사아자차카타파하고노도로모보소오조초코토포호구누두루무부수우주추")
GOODS = ["커피", "카페서비스업", "김치", "식당업", "과자", "맥주", "국수", "도시락"]
LEXICON = {"PRODUCT": ["커피", "김치", "국수"], "SERVICE_CONTENT": ["카페"], "LOCATION": ["제주"]}


@pytest.fixture(scope="session")
def analyzer():
    return NameAnalyzer(cmudict={"david": "deibid", "coffee": "kapi"}, generic_terms=["커피", "카페"])


def _name(rng) -> str:
    return "".join(rng.choice(SYLLABLES, size=rng.integers(3, 6)))


def make_pool(rng, count: int = 150) -> pd.DataFrame:
    rows = []
    for i in range(count):
        filed = 20100101 + int(rng.integers(0, 6)) * 10000 + int(rng.integers(0, 12)) * 100
        rows.append(
            {
                "app": f"40{2010000000 + i}",
                "name": _name(rng),
                "date": str(filed),
                "출원공고일자": f"{str(filed + 10000)[:4]}.{str(filed)[4:6]}.15",
                "지정상품": "|".join(rng.choice(GOODS, size=2, replace=False)),
                "classes": str(rng.choice([29, 30, 32, 43])),
                "유사군코드": "G0101",
                "원출원번호": "",
            }
        )
    return pd.DataFrame(rows)


def make_cohort(rng, pool: pd.DataFrame) -> pd.DataFrame:
    """선행상표와 이름·상품이 겹치면 거절(1)이 되도록 만든 단순한 세계."""
    periods = [
        ("fit", 2019, 240),
        ("calibration", 2022, 80),
        ("validation", 2023, 80),
        ("test", 2024, 60),
        ("time_2025", 2025, 40),
        ("time_2026", 2026, 40),
    ]
    rows, used = [], set(pool["name"])
    for period, year, count in periods:
        for i in range(count):
            refused = i % 2
            if refused:
                prior = pool.iloc[int(rng.integers(len(pool)))]
                name, goods = prior["name"] + rng.choice(SYLLABLES), prior["지정상품"]
            else:
                name, goods = _name(rng) + _name(rng), "|".join(rng.choice(GOODS, size=2, replace=False))
            while name in used:
                name += rng.choice(SYLLABLES)
            used.add(name)
            app = f"40{year}{len(rows):06d}"
            rows.append(
                dict(
                    app=app,
                    name=name,
                    goods=goods,
                    classes="30|43",
                    logo=float(i % 3 == 0),
                    date=f"{year}0601",
                    y=refused,
                    period=period,
                    source_group=app,
                )
            )
    return pd.DataFrame(rows)


def make_text_bundle(cohort: pd.DataFrame) -> dict:
    fit = cohort[cohort["period"].eq("fit")]
    names = [normalize(n) for n in fit["name"]]
    documents = [names, [to_jamo(n) for n in names], [" ".join(sorted(goods_terms(g))) for g in fit["goods"]]]
    vectorizers = [
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False, sublinear_tf=True),
        TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, sublinear_tf=True),
        TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, sublinear_tf=True),
    ]
    for vectorizer, texts in zip(vectorizers, documents):
        vectorizer.fit(texts)
    columns = list(structure_features(cohort.iloc[0].to_dict(), LEXICON))
    return dict(vectorizers=vectorizers, numeric_columns=columns, lexicon=LEXICON, offset=sum(len(v.vocabulary_) for v in vectorizers))


@pytest.fixture(scope="session")
def workspace(tmp_path_factory) -> Paths:
    """작업 폴더 한 벌: 풀, 라벨 표, 텍스트 묶음, 공고일 레지스트리."""
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("workspace")
    paths = Paths(workspace=root, output=root / "artifacts")
    for path in (*paths.pool_tables, paths.cohort, paths.text_bundle):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.output.mkdir()

    pool = make_pool(rng)
    pool.loc[0, "출원공고일자"] = ""  # 공고일이 비어 검색에서 빠지는 선행상표
    pool.iloc[:120].to_parquet(paths.pool_tables[0], index=False)
    pool.iloc[120:].to_parquet(paths.pool_tables[1], index=False)
    cohort = make_cohort(rng, pool)
    cohort.to_parquet(paths.cohort, index=False)
    joblib.dump(make_text_bundle(cohort), paths.text_bundle)

    registry = pd.DataFrame([dict(app=pool.loc[0, "app"], publication="20170315", source_type="KIPRIS_LIVE_BIBLIOGRAPHY")])
    registry.to_csv(paths.date_registry, index=False)
    return paths

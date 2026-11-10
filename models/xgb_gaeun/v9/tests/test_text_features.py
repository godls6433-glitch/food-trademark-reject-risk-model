import pytest

from trademark_risk.text_features import structure_features

LEXICON = {"PRODUCT": ["커피"], "LOCATION": ["제주"]}


def test_context_requires_the_word_in_the_designated_goods():
    query = dict(name="제주커피", goods="커피|카페서비스업", classes="30|43", logo=1)
    features = structure_features(query, LEXICON)
    assert features["kb_PRODUCT_name"] == features["kb_PRODUCT_context"] == 0.5  # '커피' 2/4자, 상품에도 있음
    assert features["kb_LOCATION_name"] == 0.5 and features["kb_LOCATION_context"] == 0.0  # '제주'는 상품에 없음
    assert features["kb_PRODUCT_context_30"] == 0.5 and features["kb_PRODUCT_context_29"] == 0.0
    assert features["kb_unmatched"] == 0.0 and features["logo"] == 1


def test_overlapping_matches_are_not_double_counted():
    features = structure_features(dict(name="커피커피", goods="", classes="30", logo=0), {"PRODUCT": ["커피", "피커"]})
    assert features["kb_PRODUCT_name"] == pytest.approx(1.0)

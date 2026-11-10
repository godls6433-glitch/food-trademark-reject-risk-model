from trademark_risk.textnorm import char_ngrams, goods_terms, normalize, parse_classes, parse_date


def test_normalize_keeps_only_letters_and_digits():
    assert normalize(" LIKE-HUR ") == "likehur"
    assert normalize("북위35°") == "북위35"
    assert normalize(None) == ""


def test_goods_terms_splits_on_every_separator():
    assert goods_terms("커피, 카페업|과자;\n커피") == {"커피", "카페업", "과자"}


def test_short_text_is_its_own_ngram():
    assert char_ngrams("가", 2) == {"가"}
    assert char_ngrams("", 2) == frozenset()
    assert char_ngrams("가나다", 2) == {"가나", "나다"}


def test_classes_outside_nice_range_are_ignored():
    assert parse_classes("30|43|99") == {30, 43}


def test_dates_are_validated_not_guessed():
    assert parse_date("2024.10.11") == "20241011"
    assert parse_date("20200231") == ""  # 달력에 없는 날짜
    assert parse_date("2024-10") == ""

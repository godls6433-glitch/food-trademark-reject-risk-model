from trademark_risk.phonetic import NameAnalyzer, load_cmudict


def test_korean_and_english_spellings_meet_in_one_sound_code(analyzer):
    assert analyzer.phonetic("DAVID") == analyzer.phonetic("데이빗") == "teipit"


def test_spaced_latin_letters_are_also_tried_joined(analyzer):
    assert analyzer.parts("O MI DANG") == ("omidang", "mi", "dang")


def test_generic_terms_are_stripped_from_the_edges(analyzer):
    assert analyzer.parts("제주푸른커피") == ("제주푸른커피", "제주푸른")
    assert analyzer.parts("커피") == ("커피",)  # 전체 이름 자체는 보통명칭이어도 남긴다


def test_cmudict_keeps_first_pronunciation(tmp_path):
    path = tmp_path / "cmudict.dict"
    path.write_text("read R IY1 D\nread(2) R EH1 D\n", encoding="utf-8")
    assert load_cmudict(path) == {"read": "rid"}


def test_analyzer_pickles_without_its_cache(analyzer):
    import pickle

    analyzer.phonetic("데이빗")
    clone = pickle.loads(pickle.dumps(analyzer))
    assert clone._phonetic_cache == {} and clone.phonetic("데이빗") == "teipit"
    assert isinstance(clone, NameAnalyzer)

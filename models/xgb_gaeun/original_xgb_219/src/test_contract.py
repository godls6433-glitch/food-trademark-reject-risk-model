import unittest
from features import HistoricalIndex, record, static_features, normalize


class ContractTests(unittest.TestCase):
    def test_past_only_and_same_class(self):
        idx = HistoricalIndex()
        idx.add(record("가은커피", "43", "A", 10, 0, "pool:1"))
        idx.add(record("가은커피", "29", "B", 10, 0, "pool:2"))
        q = record("가은커피", "43", "C", 20, 0, "query")
        f, top = idx.query(q)
        self.assertEqual(top[0][1]["application"], "A")
        self.assertEqual(len(top), 1)
        self.assertEqual(f["prior_similarity_1"], 1.)

    def test_same_day_future_rejected(self):
        idx = HistoricalIndex()
        idx.add(record("가은커피", "43", "A", 20, 0, "pool:1"))
        for day in (19, 20):
            with self.assertRaises(AssertionError):
                idx.query(record("가은커피", "43", "C", day, 0, "query"))

    def test_same_application_excluded_and_duplicate_prior_collapsed(self):
        idx = HistoricalIndex()
        for key, app in (("1", "A"), ("2", "A"), ("3", "B")):
            idx.add(record("가은커피", "43", app, 10, 0, key))
        _, top = idx.query(record("가은커피", "43", "B", 20, 0, "query"))
        self.assertEqual(len(top), 1)
        self.assertEqual(top[0][1]["application"], "A")

    def test_missing_name_preserved(self):
        idx = HistoricalIndex()
        f, top = idx.query(record("", "43", "A", 10, float("nan"), "query"))
        self.assertEqual(top, [])
        self.assertEqual(f["prior_similarity_1"], 0)
        self.assertEqual(static_features("", (43,), float("nan"))["name_missing"], 1)

    def test_no_label_or_status_feature(self):
        f = static_features("한우마을", (29, 43), 1.)
        self.assertFalse(set(f).intersection({"registered", "fold", "status", "등록일자", "법적상태", "출원번호"}))
        self.assertEqual(normalize("ＡＢＣ (카페)"), "abc카페")


if __name__ == "__main__":
    unittest.main()

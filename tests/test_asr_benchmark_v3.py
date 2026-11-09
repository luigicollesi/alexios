"""Unidade: métricas offline ASR, sem modelos ou WAVs de pessoas reais."""
import unittest

from scripts.benchmark_asr_v3 import edits, phrase_count, summarize, words


class AsrBenchmarkTests(unittest.TestCase):
    def test_wer_cer_and_name_recall_on_luigi_misheard_as_muita(self):
        records = [{
            "reference": "Luigi",
            "predicted": "Muita!",
            "elapsed_ms": 20.0,
        }]
        result = summarize(records, ("Luigi",))
        self.assertEqual(result["wer"], 1.0)
        self.assertGreater(result["cer"], 0)
        self.assertEqual(result["name_title_reference_mentions"], 1)
        self.assertEqual(result["name_title_recall"], 0.0)

    def test_correct_name_but_false_inserted_title(self):
        result = summarize([{
            "reference": "Luigi falou",
            "predicted": "Luigi Família Mazzanti falou",
            "elapsed_ms": 10.0,
        }], ("Luigi", "Família Mazzanti"))
        self.assertEqual(result["name_title_recall"], 1.0)
        self.assertEqual(result["name_title_false_insertions"], 1)
        self.assertGreater(result["wer"], 0.0)

    def test_repeated_terms_are_counted_as_real_occurrences(self):
        self.assertEqual(phrase_count(words("Luigi, Luigi"), words("Luigi")), 2)
        report = summarize([{
            "reference": "Luigi, Luigi",
            "predicted": "Luigi",
            "elapsed_ms": 20.0,
        }], ("Luigi",))
        self.assertEqual(report["name_title_recall"], 0.5)

    def test_empty_and_unicode_inputs(self):
        self.assertEqual(words("Família Mazzanti"), ["família", "mazzanti"])
        self.assertEqual(edits("muita", "luigi"), 4)
        empty = summarize([], ("Luigi",))
        self.assertIsNone(empty["wer"])
        self.assertIsNone(empty["cer"])
        self.assertIsNone(empty["name_title_recall"])


if __name__ == "__main__":
    unittest.main()

"""Pure metrics for the offline speaker benchmark."""
import unittest

from scripts.evaluate_speakers import summarize


class SpeakerEvaluationTests(unittest.TestCase):
    def test_summary_flags_fragmentation_and_false_merges(self):
        rows = [
            {"speaker": "A", "predicted": "speaker_01", "latency_ms": 10},
            {"speaker": "A", "predicted": "speaker_02", "latency_ms": 20},
            {"speaker": "B", "predicted": "speaker_02", "latency_ms": 30},
            {"speaker": "B", "predicted": "speaker_unknown", "latency_ms": 40},
        ]
        result = summarize(rows)
        self.assertEqual(result["fragmentation_extra_ids"], 1)
        self.assertEqual(result["false_merge_extra_people"], 1)
        self.assertEqual(result["unknown_rate"], 0.25)
        self.assertEqual(result["latency_p50_ms"], 25.0)
        self.assertEqual(result["turns"], 4)

    def test_empty_manifest_is_safe_for_summary(self):
        result = summarize([])
        self.assertEqual(result["turns"], 0)
        self.assertEqual(result["fragmentation_extra_ids"], 0)
        self.assertEqual(result["latency_p95_ms"], 0)


if __name__ == "__main__":
    unittest.main()

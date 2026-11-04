import unittest
import numpy as np

from audio_windows import WindowCutter, SAMPLE_RATE


class WindowCutterTests(unittest.TestCase):
    def test_four_second_windows_have_one_second_overlap(self):
        cutter = WindowCutter()
        signal = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
        results = []
        for frame in np.array_split(signal, 40):
            results.extend(cutter.push(frame))
        self.assertEqual([w.start_ms for w in results], [0, 3000, 6000])
        self.assertEqual([w.index for w in results], [0, 1, 2])
        self.assertTrue(np.array_equal(results[0].samples[3 * SAMPLE_RATE:],
                                       results[1].samples[:SAMPLE_RATE]))
        self.assertTrue(np.array_equal(results[1].samples[3 * SAMPLE_RATE:],
                                       results[2].samples[:SAMPLE_RATE]))

    def test_stops_before_four_seconds(self):
        cutter = WindowCutter()
        self.assertEqual(cutter.push(np.ones(3 * SAMPLE_RATE, dtype=np.float32)), [])
        self.assertEqual(len(cutter.push(np.ones(SAMPLE_RATE, dtype=np.float32))), 1)


if __name__ == "__main__":
    unittest.main()

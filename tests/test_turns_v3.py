"""Testes da máquina de turnos, sem baixar modelos nem usar microfone."""

import unittest
import numpy as np

from agent.v3.turns import (
    SAMPLE_RATE, CapturedTurn, FloorOpened, SpeechStarted, TurnEngine,
)

FRAME = np.full(512, 0.1, dtype=np.float32)
SILENCE = np.zeros(512, dtype=np.float32)


class StubClassifier:
    def __init__(self, complete=True):
        self.complete_result = complete
        self.calls = 0

    def complete(self, audio):
        self.calls += 1
        self.last_len = len(audio)
        return self.complete_result


class TurnEngineTests(unittest.TestCase):
    def frames(self, eng, n, speaking):
        emitted = []
        for _ in range(n):
            emitted.extend(eng.feed(FRAME if speaking else SILENCE, speaking))
        return emitted

    def test_complete_turn_then_one_floor_open(self):
        classifier = StubClassifier(complete=True)
        engine = TurnEngine(classifier)
        events = self.frames(engine, 20, True)
        self.assertEqual(len([e for e in events if isinstance(e, SpeechStarted)]), 1)
        events += self.frames(engine, 45, False)
        turns = [e for e in events if isinstance(e, CapturedTurn)]
        floors = [e for e in events if isinstance(e, FloorOpened)]
        self.assertEqual(len(turns), 1)
        self.assertTrue(turns[0].complete)
        self.assertEqual(len(floors), 1)
        self.assertEqual(floors[0].activity_epoch, 1)
        self.assertLessEqual(classifier.last_len, SAMPLE_RATE * 8)
        self.assertEqual(self.frames(engine, 100, False), [])

    def test_incomplete_pause_keeps_same_turn(self):
        classifier = StubClassifier(complete=False)
        engine = TurnEngine(classifier)
        events = self.frames(engine, 20, True)
        events += self.frames(engine, 20, False)  # 640 ms de hesitação
        self.assertFalse(any(isinstance(e, (CapturedTurn, FloorOpened)) for e in events))
        events += self.frames(engine, 15, True)
        self.assertFalse(any(isinstance(e, CapturedTurn) for e in events))
        events += self.frames(engine, 85, False)  # timeout de pausa, fallback
        self.assertEqual(len([e for e in events if isinstance(e, CapturedTurn)]), 1)

    def test_floor_invalidated_when_new_voice_begins(self):
        engine = TurnEngine(StubClassifier(complete=True), initial_epoch=9)
        self.frames(engine, 20, True)
        self.frames(engine, 50, False)
        restarted = self.frames(engine, 2, True)
        starts = [e for e in restarted if isinstance(e, SpeechStarted)]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].activity_epoch, 11)

    def test_long_turn_splits_without_fake_completion(self):
        engine = TurnEngine(StubClassifier(), max_segment_s=5)
        events = self.frames(engine, 200, True)
        partials = [e for e in events if isinstance(e, CapturedTurn)]
        self.assertGreaterEqual(len(partials), 1)
        self.assertTrue(all(not e.complete for e in partials))
        self.assertFalse(any(isinstance(e, FloorOpened) for e in events))
        self.assertLessEqual(max(len(e.audio) for e in partials), SAMPLE_RATE * 5 + len(FRAME))

    def test_stop_flushes_partial_without_opening_floor(self):
        engine = TurnEngine(StubClassifier(complete=False))
        self.frames(engine, 20, True)
        fragment = engine.flush_partial()
        self.assertIsNotNone(fragment)
        self.assertFalse(fragment.complete)
        self.assertGreater(len(fragment.audio), 0)
        self.assertIsNone(engine.flush_partial())


    def test_speaker_audio_contains_vad_positive_frames_not_trailing_silence(self):
        engine = TurnEngine(StubClassifier(complete=True))
        self.frames(engine, 30, True)  # 960 ms of voice
        events = self.frames(engine, 45, False)
        turns = [item for item in events if isinstance(item, CapturedTurn)]
        self.assertEqual(len(turns), 1)
        turn = turns[0]
        self.assertIsNotNone(turn.speaker_audio)
        self.assertEqual(len(turn.speaker_audio), 30 * len(FRAME))
        self.assertGreater(len(turn.audio), len(turn.speaker_audio))
        self.assertTrue(np.all(turn.speaker_audio == 0.1))

    def test_speaker_audio_is_bounded_to_eight_seconds_and_resets_on_new_turn(self):
        engine = TurnEngine(StubClassifier(complete=True), max_segment_s=20)
        self.frames(engine, 320, True)  # 10.24 s
        events = self.frames(engine, 45, False)
        turns = [item for item in events if isinstance(item, CapturedTurn)]
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0].speaker_audio), 8 * SAMPLE_RATE)
        self.frames(engine, 25, True)
        events = self.frames(engine, 45, False)
        turns = [item for item in events if isinstance(item, CapturedTurn)]
        self.assertEqual(len(turns), 1)
        self.assertEqual(len(turns[0].speaker_audio), 25 * len(FRAME))

    def test_no_floor_without_prior_speech(self):
        engine = TurnEngine(StubClassifier())
        self.assertEqual(self.frames(engine, 200, False), [])


if __name__ == "__main__":
    unittest.main()

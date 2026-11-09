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


    def test_preroll_is_192_milliseconds_not_192_seconds(self):
        engine = TurnEngine(StubClassifier(complete=True), preroll_ms=192)
        self.assertEqual(engine.preroll_samples, 3072)
        self.frames(engine, 800, False)  # 25.6 seconds of ambient audio
        self.assertEqual(engine.preroll_length, 3072)
        self.assertLessEqual(len(engine.preroll), 6)
        events = self.frames(engine, 28, True)
        events += self.frames(engine, 45, False)
        turns = [e for e in events if isinstance(e, CapturedTurn)]
        self.assertEqual(len(turns), 1)
        self.assertGreater(turns[0].started_ms, 25_000)
        self.assertLess(len(turns[0].audio), 2 * SAMPLE_RATE)

    def test_separate_turns_never_replay_previous_voice_or_restart_at_zero(self):
        """Reproduz regressão de start_ms=0 e áudio de 5/19/36 segundos."""
        engine = TurnEngine(StubClassifier(complete=True))
        turns = []
        for n, magnitude in ((32, 0.11), (65, 0.22), (90, 0.33)):
            frame = np.full(512, magnitude, dtype=np.float32)
            self.frames(engine, 100, False)
            for _ in range(n):
                engine.feed(frame, True)
            events = self.frames(engine, 45, False)
            newest = [e for e in events if isinstance(e, CapturedTurn)]
            self.assertEqual(len(newest), 1)
            turns.append(newest[0])
        self.assertEqual(len(turns), 3)
        self.assertTrue(turns[0].started_ms > 0)
        for i in (1, 2):
            self.assertGreater(turns[i].started_ms, turns[i - 1].ended_ms)
        for turn, (n, magnitude) in zip(turns, ((32, 0.11), (65, 0.22), (90, 0.33))):
            self.assertLessEqual(len(turn.audio), (n + 20) * 512)
            self.assertTrue(np.any(np.isclose(turn.audio, magnitude)))
            self.assertEqual(len(turn.speaker_audio), n * 512)
        self.assertFalse(np.any(np.isclose(turns[1].audio, 0.11)))
        self.assertFalse(np.any(np.isclose(turns[2].audio, 0.22)))

    def test_a_stale_cumulative_buffer_is_trimmed_to_this_turn_only(self):
        """Safeguard even if an upstream regression leaves old audio frames."""
        engine = TurnEngine(StubClassifier(complete=True))
        old = np.full(512, 0.27, dtype=np.float32)
        new = np.full(512, 0.53, dtype=np.float32)
        for _ in range(30):
            engine.feed(old, True)
        first_events = self.frames(engine, 45, False)
        self.assertEqual(len([e for e in first_events if isinstance(e, CapturedTurn)]), 1)
        self.frames(engine, 30, False)
        # Reproduce a corrupted retained buffer. Bounds from VAD frame clock
        # still ensure Whisper gets only newly voiced audio.
        engine.turn_parts = [np.full(45 * 512, 0.27, dtype=np.float32)]
        engine.turn_length = 45 * 512
        for _ in range(35):
            engine.feed(new, True)
        next_events = self.frames(engine, 45, False)
        turns = [e for e in next_events if isinstance(e, CapturedTurn)]
        self.assertEqual(len(turns), 1)
        self.assertFalse(np.any(np.isclose(turns[0].audio, 0.27)))
        self.assertTrue(np.any(np.isclose(turns[0].audio, 0.53)))
        self.assertGreater(turns[0].started_ms, 0)

    def test_no_floor_without_prior_speech(self):
        engine = TurnEngine(StubClassifier())
        self.assertEqual(self.frames(engine, 200, False), [])


if __name__ == "__main__":
    unittest.main()

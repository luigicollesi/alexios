"""Teste dos eventos ordenados e cancelamento sem rede real."""
import threading
import time
import unittest

import numpy as np

from agent.v3.dispatch import EventDispatcher
from agent.v3.turns import CapturedTurn, FloorOpened


class FakeClient:
    def __init__(self):
        self.events = []
        self.activities = []

    def send_event(self, kind, payload):
        self.events.append((kind, payload.copy()))

    def announce_activity(self, epoch):
        self.activities.append(epoch)


class FakeTranscriber:
    def __init__(self):
        self.unblock = threading.Event()
        self.unblock.set()
        self.sent_to_asr = threading.Event()

    def transcribe(self, audio):
        self.sent_to_asr.set()
        if not self.unblock.wait(3):
            raise RuntimeError("Test timed out")
        return "O contexto é compartilhado."


class FakeSpeakers:
    def __init__(self):
        self.counter = 0

    def identify(self, audio):
        self.counter += 1
        return "speaker_01" if self.counter == 1 else "speaker_02"


def wait_until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def turn(epoch, n):
    return CapturedTurn(
        turn_id=f"turn-{n}", audio=np.full(16000, 0.1, dtype=np.float32),
        started_ms=n * 1000, ended_ms=(n+1)*1000,
        complete=True, activity_epoch=epoch,
    )


class DispatchTests(unittest.TestCase):
    def test_every_speaker_has_same_instance_and_sequential_events(self):
        client = FakeClient()
        tx = FakeTranscriber()
        sender = EventDispatcher(client, tx, FakeSpeakers(), next_sequence=20)
        try:
            sender.speech_started(1)
            self.assertTrue(wait_until(lambda: sender.acknowledged_activity >= 1))
            sender.enqueue(turn(1, 1))
            sender.speech_started(2)
            self.assertTrue(wait_until(lambda: sender.acknowledged_activity >= 2))
            sender.enqueue(turn(2, 2))
            sender.enqueue(FloorOpened(activity_epoch=2, silence_ms=1300))
            self.assertTrue(wait_until(lambda: len(client.events) == 3))
            kinds = [x[0] for x in client.events]
            self.assertEqual(kinds, ["turn_end", "turn_end", "floor_open"])
            self.assertEqual([p["sequence"] for _, p in client.events], [20, 21, 22])
            self.assertEqual([client.events[0][1]["speakerId"],
                              client.events[1][1]["speakerId"]],
                             ["speaker_01", "speaker_02"])
            self.assertEqual(client.events[2][1]["afterSequence"], 21)
            self.assertEqual(len(client.activities), 2)
        finally:
            sender.shutdown()


    def test_whisper_gets_full_turn_and_cam_plus_only_vad_voice(self):
        client = FakeClient()
        class SpyTranscriber:
            seen = None
            def transcribe(self, audio):
                self.seen = audio.copy()
                return "Alô, tudo bem?"
        class SpySpeakers:
            seen = None
            def identify(self, audio):
                self.seen = audio.copy()
                return "speaker_unknown"
        tx, speakers = SpyTranscriber(), SpySpeakers()
        sender = EventDispatcher(client, tx, speakers, next_sequence=1)
        try:
            sender.speech_started(1)
            self.assertTrue(wait_until(lambda: sender.acknowledged_activity >= 1))
            full = np.concatenate((np.full(32000, 0.1, dtype=np.float32),
                                   np.zeros(80000, dtype=np.float32)))
            clean = full[:32000].copy()
            sender.enqueue(CapturedTurn(
                turn_id="turn-voice", audio=full, speaker_audio=clean,
                started_ms=0, ended_ms=7000, complete=True, activity_epoch=1,
            ))
            self.assertTrue(wait_until(lambda: len(client.events) == 1))
            self.assertEqual(len(tx.seen), len(full))
            self.assertEqual(len(speakers.seen), len(clean))
            self.assertEqual(client.events[0][1]["speakerId"], "speaker_unknown")
        finally:
            sender.shutdown()

    def test_empty_transcription_does_not_open_floor(self):
        client = FakeClient()
        class EmptyTranscriber:
            def transcribe(self, audio):
                return ""
        sender = EventDispatcher(client, EmptyTranscriber(), FakeSpeakers(),
                                 next_sequence=1)
        try:
            sender.speech_started(1)
            self.assertTrue(wait_until(lambda: sender.acknowledged_activity >= 1))
            sender.enqueue(turn(1, 1))
            sender.enqueue(FloorOpened(activity_epoch=1, silence_ms=1250))
            self.assertTrue(wait_until(lambda: sender.events.unfinished_tasks == 0))
            self.assertEqual(client.events, [])
        finally:
            sender.shutdown()

    def test_new_speech_invalidates_waiting_floor(self):
        client = FakeClient()
        tx = FakeTranscriber()
        tx.unblock.clear()
        sender = EventDispatcher(client, tx, FakeSpeakers(), next_sequence=1)
        try:
            sender.speech_started(1)
            sender.enqueue(turn(1, 1))
            self.assertTrue(tx.sent_to_asr.wait(2))
            sender.enqueue(FloorOpened(activity_epoch=1, silence_ms=1200))
            sender.speech_started(2)
            tx.unblock.set()
            self.assertTrue(wait_until(lambda: len(client.events) == 1))
            self.assertTrue(wait_until(lambda: sender.events.unfinished_tasks == 0))
            self.assertEqual([kind for kind, _ in client.events], ["turn_end"])
        finally:
            tx.unblock.set()
            sender.shutdown()


if __name__ == "__main__":
    unittest.main()

"""Capture worker: retry the identical window after a transient web failure.

This test stubs sounddevice; CI does not need a physical microphone.
"""
import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from agent.bridge import BridgeError


class UploadOnceAfterError:
    def __init__(self):
        self.calls = []
        self.completed = threading.Event()

    def upload(self, capture_id, index, words):
        self.calls.append((capture_id, index, list(words)))
        if len(self.calls) == 1:
            raise BridgeError("offline")
        self.completed.set()
        return len(words)


class FakeWhisper:
    def transcribe(self, samples, **kwargs):
        word = types.SimpleNamespace(word=" teste", start=0.1, end=0.4)
        return [types.SimpleNamespace(words=[word])], None


class CaptureTests(unittest.TestCase):
    def test_failed_window_upload_reuses_capture_and_index(self):
        fake_sd = types.SimpleNamespace(PortAudioError=RuntimeError)
        with patch.dict(sys.modules, {"sounddevice": fake_sd}):
            cap_module = importlib.import_module("agent.capture")

        client = UploadOnceAfterError()
        capture = cap_module.AudioCapture(client, FakeWhisper(), None)
        try:
            with capture.lock:
                capture.generation = 1
                capture.capture_id = "12345678-1234-4321-abcd-0123456789ab"
                capture.active.set()
            self.assertLessEqual(capture.queue.maxsize, 8)
            window = np.full(4 * 16000, 0.25, dtype=np.float32)
            capture.queue.put((capture.capture_id, 1, 0, 0, window))
            self.assertTrue(client.completed.wait(timeout=3))
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0], client.calls[1])
            self.assertEqual(client.calls[0][1], 0)
        finally:
            capture.shutdown()


if __name__ == "__main__":
    unittest.main()

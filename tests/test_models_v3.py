"""Adapters are exercised without downloading ONNX files or installing sherpa.

These tests validate package API wiring, not real-world model accuracy.
"""

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from agent.v3 import models


class FakeStream:
    def __init__(self):
        self.audio = None

    def accept_waveform(self, *, sample_rate, waveform):
        assert sample_rate == 16000
        self.audio = waveform

    def input_finished(self):
        pass


class FakeEmbeddingConfig:
    def __init__(self, **kwargs):
        self.params = kwargs

    def validate(self):
        return True


class FakeExtractor:
    def __init__(self, config):
        self.dim = 2

    def create_stream(self):
        return FakeStream()

    def is_ready(self, stream):
        return True

    def compute(self, stream):
        return [1.0, 0.0] if np.mean(stream.audio) >= 0 else [0.0, 1.0]


class FakeManager:
    def __init__(self, dim):
        self.people = {}

    def add(self, name, embedding):
        self.people[name] = np.asarray(embedding)
        return True

    def search(self, vector, *, threshold):
        a = np.asarray(vector)
        for name, b in self.people.items():
            if float(np.dot(a, b)) >= threshold:
                return name
        return ""


class FakeVadConfig:
    def __init__(self):
        self.silero_vad = types.SimpleNamespace(window_size=512)
        self.sample_rate = 0


class FakeVad:
    def __init__(self, config, *, buffer_size_in_seconds):
        assert config.silero_vad.max_speech_duration <= buffer_size_in_seconds
        self.buffer_size = buffer_size_in_seconds
        self.detected = False

    def accept_waveform(self, samples):
        self.detected = np.max(np.abs(samples)) > 0.01

    def is_speech_detected(self):
        return self.detected

    def empty(self):
        return True

    def pop(self):
        pass


class FakeOptions:
    def __init__(self):
        self.inter_op_num_threads = None
        self.intra_op_num_threads = None
        self.graph_optimization_level = None


class FakeSession:
    def __init__(self, path, *, sess_options, providers):
        self.path = path

    def get_inputs(self):
        return [types.SimpleNamespace(name="input_features")]

    def run(self, outputs, inputs):
        assert "input_features" in inputs
        return [np.array([[0.75]], dtype=np.float32)]


class FakeFeatureExtractor:
    def __init__(self, *, chunk_length):
        assert chunk_length == 8

    def __call__(self, samples, **kwargs):
        return types.SimpleNamespace(
            input_features=np.zeros((1, 80, 800), dtype=np.float32)
        )


class ModelAdapterTests(unittest.TestCase):
    def test_speaker_registration_requires_voice_quality_and_consistent_evidence(self):
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(models, "require_model", return_value="fake.onnx"):
            speakers = models.CamPlusRegistry("fake.onnx")
            good = np.full(32000, 0.1, dtype=np.float32)
            other = np.full(32000, -0.1, dtype=np.float32)
            short = np.full(16000, 0.1, dtype=np.float32)
            self.assertEqual(speakers.identify(short), "speaker_unknown")
            self.assertEqual(speakers.identify(good), "speaker_01")
            self.assertEqual(speakers.identify(good), "speaker_01")
            # Duas vozes claramente distintas, registradas imediatamente.
            self.assertEqual(speakers.identify(other), "speaker_02")
            self.assertEqual(speakers.last_decision["reason"], "enrolled_distinct")
            # A voltou a falar: o mesmo ID speaker_01, NUNCA speaker_03.
            self.assertEqual(speakers.identify(good), "speaker_01")
            self.assertEqual(speakers.identify(other), "speaker_02")
            self.assertEqual(speakers.identify(good), "speaker_01")
            self.assertEqual(speakers.counter, 2)
            self.assertEqual(set(speakers.references), {"speaker_01", "speaker_02"})

    def test_ambiguous_B_candidate_survives_interleaved_A_and_keeps_A_id(self):
        """A→01, B incerto→unknown, A→01, B novamente→02, A→01."""
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(
            models, "require_model", return_value="fake.onnx",
        ):
            speakers = models.CamPlusRegistry("fake.onnx")
            audio = np.full(32000, 0.1, dtype=np.float32)
            speakers.extractor.compute = lambda stream: [1.0, 0.0]
            self.assertEqual(speakers.identify(audio), "speaker_01")
            # B is different but not dissimilar enough for instant enrollment.
            speakers.extractor.compute = lambda stream: [0.40, 0.916515]
            self.assertEqual(speakers.identify(audio), "speaker_unknown")
            self.assertEqual(speakers.last_decision["reason"], "awaiting_second_sample")
            speakers.extractor.compute = lambda stream: [1.0, 0.0]
            self.assertEqual(speakers.identify(audio), "speaker_01")
            # A's turn must not erase pending enrollment of B.
            speakers.extractor.compute = lambda stream: [0.40, 0.916515]
            self.assertEqual(speakers.identify(audio), "speaker_02")
            speakers.extractor.compute = lambda stream: [1.0, 0.0]
            self.assertEqual(speakers.identify(audio), "speaker_01")
            self.assertEqual(speakers.counter, 2)

    def test_short_or_ambiguous_returning_A_does_not_create_speaker_03(self):
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(
            models, "require_model", return_value="fake.onnx",
        ):
            speakers = models.CamPlusRegistry("fake.onnx")
            good = np.full(32000, 0.1, dtype=np.float32)
            speakers.extractor.compute = lambda stream: [1.0, 0.0]
            self.assertEqual(speakers.identify(good), "speaker_01")
            speakers.extractor.compute = lambda stream: [0.0, 1.0]
            self.assertEqual(speakers.identify(good), "speaker_02")
            # A is briefly distorted, close to recognition threshold.
            speakers.extractor.compute = lambda stream: [0.58, 0.55]
            self.assertEqual(speakers.identify(good), "speaker_unknown")
            self.assertEqual(speakers.last_decision["reason"], "near_existing_unknown")
            speakers.extractor.compute = lambda stream: [1.0, 0.0]
            self.assertEqual(speakers.identify(good), "speaker_01")
            self.assertEqual(speakers.counter, 2)

    def test_short_noisy_or_clipped_voice_never_enrolls(self):
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(models, "require_model", return_value="fake.onnx"):
            speakers = models.CamPlusRegistry("fake.onnx")
            cases = [
                np.full(5000, 0.1, dtype=np.float32),
                np.zeros(32000, dtype=np.float32),
                np.full(32000, 0.0001, dtype=np.float32),
                np.ones(32000, dtype=np.float32),
                np.full(32000, np.nan, dtype=np.float32),
            ]
            for bad_audio in cases:
                self.assertEqual(speakers.identify(bad_audio), "speaker_unknown")
            self.assertEqual(speakers.counter, 0)

    def test_near_threshold_rejection_does_not_spawn_duplicate_speaker(self):
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(models, "require_model", return_value="fake.onnx"):
            speakers = models.CamPlusRegistry("fake.onnx")
            normal = np.full(32000, 0.1, dtype=np.float32)
            self.assertEqual(speakers.identify(normal), "speaker_01")
            speakers.extractor.compute = lambda stream: [0.56, 0.8285]
            self.assertEqual(speakers.identify(normal), "speaker_unknown")
            self.assertEqual(speakers.last_decision["reason"], "near_existing_unknown")
            self.assertEqual(speakers.counter, 1)

    def test_multiple_reliable_references_reduce_same_person_fragmentation(self):
        sherpa = types.SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=FakeEmbeddingConfig,
            SpeakerEmbeddingExtractor=FakeExtractor,
            SpeakerEmbeddingManager=FakeManager,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(models, "require_model", return_value="fake.onnx"):
            speakers = models.CamPlusRegistry("fake.onnx")
            samples = np.full(32000, 0.1, dtype=np.float32)
            self.assertEqual(speakers.identify(samples), "speaker_01")
            # Mesmo locutor numa segunda gravação ainda reconhecida pelo manager.
            speakers.extractor.compute = lambda stream: [0.8, 0.6]
            self.assertEqual(speakers.identify(samples), "speaker_01")
            self.assertEqual(len(speakers.references["speaker_01"]), 2)
            # A referência original não reconheceria esta terceira variação,
            # mas a segunda amostra de referência é altamente semelhante.
            speakers.extractor.compute = lambda stream: [0.3, 0.95394]
            self.assertEqual(speakers.identify(samples), "speaker_01")
            self.assertEqual(speakers.last_decision["reason"], "gallery_recovered")
            self.assertEqual(speakers.counter, 1)

    def test_vad_uses_512_samples_and_reports_voice(self):
        sherpa = types.SimpleNamespace(
            VadModelConfig=FakeVadConfig,
            VoiceActivityDetector=FakeVad,
        )
        with patch.dict(sys.modules, {"sherpa_onnx": sherpa}), patch.object(models, "require_model", return_value="fake.onnx"):
            vad = models.SileroVad("fake.onnx")
            self.assertEqual(vad.window_size, 512)
            self.assertGreaterEqual(vad.detector.buffer_size, 100)
            self.assertFalse(vad.is_speech(np.zeros(512, dtype=np.float32)))
            self.assertTrue(vad.is_speech(np.ones(512, dtype=np.float32)))
            with self.assertRaises(ValueError):
                vad.is_speech(np.ones(128, dtype=np.float32))

    def test_smart_turn_expects_probability_from_onnx(self):
        ort = types.SimpleNamespace(
            SessionOptions=FakeOptions,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=1),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=2),
            InferenceSession=FakeSession,
        )
        transformers = types.SimpleNamespace(WhisperFeatureExtractor=FakeFeatureExtractor)
        with patch.dict(sys.modules, {"onnxruntime": ort, "transformers": transformers}), patch.object(models, "require_model", return_value="fake.onnx"):
            turn = models.SmartTurnDetector("fake.onnx", threshold=0.5)
            self.assertTrue(turn.complete(np.ones(16000, dtype=np.float32)))
            turn.threshold = 0.9
            self.assertFalse(turn.complete(np.ones(16000, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()

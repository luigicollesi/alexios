"""Regressões de transcrição contextual, sem baixar o Whisper real."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from agent.v3.transcription import (
    WhisperTurnTranscriber, load_vocabulary,
)


class FakeWhisper:
    def __init__(self, text="Muita!", logprob=-0.42):
        self.calls = []
        self.text = text
        self.logprob = logprob

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples.copy(), kwargs))
        return iter([SimpleNamespace(text=self.text, avg_logprob=self.logprob)]), None


class TranscriptionV3Tests(unittest.TestCase):
    def sample(self):
        return np.full(16000, 0.1, dtype=np.float32)

    def test_hotwords_reaches_whisper_without_forcing_a_name(self):
        model = FakeWhisper("Muita!")
        stt = WhisperTurnTranscriber(
            model, vocabulary=("Luigi", "Alexios"), context_mode="hotwords",
        )
        self.assertEqual(stt.transcribe(self.sample()), "Muita!")
        _, options = model.calls[0]
        self.assertEqual(options["hotwords"], "Luigi, Alexios")
        self.assertNotIn("initial_prompt", options)
        self.assertEqual(options["language"], "pt")
        self.assertEqual(options["beam_size"], 5)
        self.assertFalse(options["vad_filter"])
        self.assertFalse(options["condition_on_previous_text"])
        self.assertEqual(stt.last_diagnostics["mean_avg_logprob"], -0.42)

    def test_initial_prompt_is_an_alternative_not_an_additional_bias(self):
        model = FakeWhisper("Luigi")
        stt = WhisperTurnTranscriber(
            model, vocabulary=("Luigi",), context_mode="prompt", beam_size=3,
            vad_filter=True,
        )
        self.assertEqual(stt.transcribe(self.sample()), "Luigi")
        _, options = model.calls[0]
        self.assertEqual(options["initial_prompt"], "Luigi")
        self.assertNotIn("hotwords", options)
        self.assertEqual(options["beam_size"], 3)
        self.assertTrue(options["vad_filter"])

    def test_none_mode_has_no_bias(self):
        model = FakeWhisper("muita")
        stt = WhisperTurnTranscriber(model, vocabulary=("Luigi",), context_mode="none")
        self.assertEqual(stt.transcribe(self.sample()), "muita")
        self.assertNotIn("hotwords", model.calls[0][1])
        self.assertNotIn("initial_prompt", model.calls[0][1])

    def test_real_repeated_words_are_not_deduplicated(self):
        model = FakeWhisper("Luigi, Luigi!")
        self.assertEqual(
            WhisperTurnTranscriber(model, vocabulary=("Luigi",)).transcribe(self.sample()),
            "Luigi, Luigi!",
        )

    def test_separate_turns_do_not_inherit_previous_text(self):
        model = FakeWhisper("Bom dia")
        stt = WhisperTurnTranscriber(model, vocabulary=("Luigi",))
        self.assertEqual(stt.transcribe(self.sample()), "Bom dia")
        model.text = "Luigi"
        self.assertEqual(stt.transcribe(self.sample()), "Luigi")
        self.assertTrue(all(
            not opts["condition_on_previous_text"] for _, opts in model.calls
        ))
        self.assertNotIn("Bom dia", str(model.calls[1][1]))

    def test_silence_and_corrupted_waveform_skip_model(self):
        model = FakeWhisper()
        stt = WhisperTurnTranscriber(model)
        self.assertEqual(stt.transcribe(np.zeros(16000, dtype=np.float32)), "")
        self.assertEqual(stt.transcribe(np.full(16000, np.nan, dtype=np.float32)), "")
        self.assertEqual(model.calls, [])

    def test_invalid_configs_raise(self):
        model = FakeWhisper()
        for kwargs in (
            {"beam_size": 0}, {"beam_size": 9},
            {"context_mode": "magic"}, {"vocabulary": ("x" * 400,)},
        ):
            with self.assertRaises(ValueError):
                WhisperTurnTranscriber(model, **kwargs)

    def test_vocabulary_json_validated_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "names.json"
            path.write_text(
                json.dumps({"terms": ["Luigi", "luigi", "Alexios", "Família Mazzanti"]}),
                encoding="utf-8",
            )
            self.assertEqual(
                load_vocabulary(path),
                ("Luigi", "Alexios", "Família Mazzanti"),
            )
            for malformed in (
                {"terms": "Luigi"},
                {"terms": ["ok", "\n\n"]},
                {"terms": ["x" * 65]},
                {"terms": ["foo, bar"]},
                {"terms": ["ok"], "injection": "never"},
                {"terms": [str(i) for i in range(30)]},
            ):
                path.write_text(json.dumps(malformed), encoding="utf-8")
                with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                    load_vocabulary(path)

    def test_default_vocabulary_includes_names_and_titles(self):
        vocab = load_vocabulary(
            Path(__file__).resolve().parent.parent / "config" / "asr-vocabulary.json"
        )
        for item in ("Luigi", "Alexios", "Família Mazzanti"):
            self.assertIn(item, vocab)


if __name__ == "__main__":
    unittest.main()

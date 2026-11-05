"""Transcrição local por turnos concluídos ou segmentos parciais."""

from __future__ import annotations

import numpy as np


class WhisperTurnTranscriber:
    def __init__(self, model: object) -> None:
        self.model = model

    def transcribe(self, samples: np.ndarray) -> str:
        if not len(samples) or float(np.max(np.abs(samples))) < 0.002:
            return ""
        segments, _ = self.model.transcribe(
            np.ascontiguousarray(samples, dtype=np.float32),
            language="pt",
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments
                        if segment.text.strip()).strip()

"""Transcrição local por turno com vocabulário contextual opcional.

Os termos são pistas acústico-linguísticas, nunca correções automáticas.
O objeto não memoriza texto de turnos anteriores. Modelos e áudio não
são enviados para serviços externos por este módulo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

SAMPLE_RATE = 16_000
MAX_TERMS = 24
MAX_TERM_CHARS = 64
MAX_CONTEXT_CHARS = 360


def load_vocabulary(path: str | Path) -> tuple[str, ...]:
    """Carrega uma lista curta e validada; falha explicitamente se inválida."""
    source = Path(path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"terms"}:
        raise ValueError("Vocabulário ASR deve ser um objeto JSON com apenas 'terms'.")
    terms = data["terms"]
    if not isinstance(terms, list) or len(terms) > MAX_TERMS:
        raise ValueError(f"Vocabulário ASR exige lista de até {MAX_TERMS} termos.")
    clean: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if (not isinstance(term, str) or not term.strip()
                or len(term) > MAX_TERM_CHARS
                or any(ord(char) < 32 for char in term)):
            raise ValueError("Termo ASR inválido: use texto curto sem controles.")
        item = " ".join(term.strip().split())
        if "," in item or ";" in item:
            raise ValueError("Termos ASR não devem conter vírgulas ou ponto e vírgula.")
        if item.casefold() not in seen:
            clean.append(item)
            seen.add(item.casefold())
    if len(", ".join(clean)) > MAX_CONTEXT_CHARS:
        raise ValueError("Vocabulário ASR excedeu o limite de contexto.")
    return tuple(clean)


class WhisperTurnTranscriber:
    """ASR isolado: contexto por grafias, sem impor palavras ao resultado."""

    def __init__(
        self, model: object, *, vocabulary: Sequence[str] = (),
        context_mode: str = "hotwords", beam_size: int = 5,
    ) -> None:
        if context_mode not in {"none", "hotwords", "prompt"}:
            raise ValueError("Modo de contexto ASR inválido.")
        if not 1 <= beam_size <= 8:
            raise ValueError("beam_size ASR deve estar entre 1 e 8.")
        self.model = model
        self.context_mode = context_mode
        self.beam_size = beam_size
        self.vocabulary = tuple(vocabulary)
        if len(self.vocabulary) > MAX_TERMS or len(", ".join(self.vocabulary)) > MAX_CONTEXT_CHARS:
            raise ValueError("Vocabulário ASR muito extenso.")
        self.last_diagnostics: dict[str, float | int] = {}

    def transcribe(self, samples: np.ndarray) -> str:
        voice = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(voice) or not np.isfinite(voice).all() or float(np.max(np.abs(voice))) < 0.002:
            self.last_diagnostics = {"segments": 0, "audio_seconds": round(len(voice) / SAMPLE_RATE, 3)}
            return ""
        options = dict(
            language="pt",
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        if self.vocabulary and self.context_mode != "none":
            # Both modes are alternatives; repeating the terms in two prompt
            # channels can increase hallucinations of the suggested names.
            terms = ", ".join(self.vocabulary)
            if self.context_mode == "hotwords":
                options["hotwords"] = terms
            else:
                options["initial_prompt"] = terms

        segments, _ = self.model.transcribe(
            np.ascontiguousarray(voice), **options,
        )
        parts: list[str] = []
        log_probs: list[float] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                parts.append(text)
                score = getattr(segment, "avg_logprob", None)
                if score is not None and np.isfinite(score):
                    log_probs.append(float(score))
        self.last_diagnostics = {
            "segments": len(parts),
            "audio_seconds": round(len(voice) / SAMPLE_RATE, 3),
        }
        if log_probs:
            # Diagnostic only, not an absolute confidence probability.
            self.last_diagnostics["mean_avg_logprob"] = round(
                sum(log_probs) / len(log_probs), 3,
            )
        return " ".join(parts).strip()

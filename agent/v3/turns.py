"""Detecção de turnos baseada em VAD + decisão semântica (injetáveis/testáveis).

O áudio é acumulado apenas localmente. Fim de turno NÃO autoriza falar.
Uma abertura de fala só é emitida durante silêncio contínuo após um turno.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import numpy as np

SAMPLE_RATE = 16_000


class TurnClassifier(Protocol):
    def complete(self, recent_audio: np.ndarray) -> bool: ...


@dataclass(frozen=True)
class SpeechStarted:
    activity_epoch: int


@dataclass(frozen=True)
class CapturedTurn:
    turn_id: str
    audio: np.ndarray
    started_ms: int
    ended_ms: int
    complete: bool
    activity_epoch: int


@dataclass(frozen=True)
class FloorOpened:
    activity_epoch: int
    silence_ms: int


LocalEvent = SpeechStarted | CapturedTurn | FloorOpened


class TurnEngine:
    """Máquina de estados determinística que recebe blocos PCM e status do VAD.

    Não executa rede nem Whisper. Os tempos derivam de contagem de amostras,
    evitando relógio de parede que pode avançar durante inferências lentas.
    """

    def __init__(
        self,
        classifier: TurnClassifier,
        *,
        min_pause_ms: int = 320,
        recheck_ms: int = 450,
        max_pause_ms: int = 2600,
        floor_silence_ms: int = 1200,
        max_segment_s: int = 45,
        preroll_ms: int = 192,
        initial_epoch: int = 0,
    ) -> None:
        if not (0 < min_pause_ms < max_pause_ms):
            raise ValueError("Intervalos de pausa inválidos.")
        if floor_silence_ms < min_pause_ms or recheck_ms <= 0 or max_segment_s < 5:
            raise ValueError("Limites da detecção inválidos.")
        self.classifier = classifier
        self.min_pause = min_pause_ms * SAMPLE_RATE // 1000
        self.recheck = recheck_ms * SAMPLE_RATE // 1000
        self.max_pause = max_pause_ms * SAMPLE_RATE // 1000
        self.floor_silence = floor_silence_ms * SAMPLE_RATE // 1000
        self.max_samples = max_segment_s * SAMPLE_RATE
        self.preroll_samples = preroll_ms * SAMPLE_RATE
        self.clock = 0
        self.activity_epoch = initial_epoch
        self.has_spoken = False
        self.floor_sent = False
        self.voiced = False
        self.silence_samples = 0
        self.last_check = 0
        self.turn_start = 0
        self.turn_parts: list[np.ndarray] = []
        self.turn_length = 0
        self.preroll: deque[np.ndarray] = deque()
        self.preroll_length = 0

    @property
    def is_speaking(self) -> bool:
        return self.voiced

    @property
    def has_open_turn(self) -> bool:
        return bool(self.turn_parts)

    def _remember_preroll(self, frame: np.ndarray) -> None:
        self.preroll.append(frame.copy())
        self.preroll_length += len(frame)
        while self.preroll and self.preroll_length > self.preroll_samples + len(frame):
            self.preroll_length -= len(self.preroll.popleft())

    def _append(self, frame: np.ndarray) -> None:
        self.turn_parts.append(frame.copy())
        self.turn_length += len(frame)

    def _audio(self) -> np.ndarray:
        return np.concatenate(self.turn_parts).astype(np.float32, copy=False)

    def _finish(self, *, complete: bool, end_sample: int) -> CapturedTurn:
        turn = CapturedTurn(
            turn_id=str(uuid4()),
            audio=self._audio(),
            started_ms=self.turn_start * 1000 // SAMPLE_RATE,
            ended_ms=end_sample * 1000 // SAMPLE_RATE,
            complete=complete,
            activity_epoch=self.activity_epoch,
        )
        self.turn_parts.clear()
        self.turn_length = 0
        self.last_check = 0
        return turn

    def flush_partial(self) -> CapturedTurn | None:
        """Finaliza apenas o buffer técnico em FIM, sem autorizar resposta."""
        if not self.turn_parts:
            return None
        return self._finish(complete=False, end_sample=self.clock)

    def feed(self, frame: np.ndarray, speech: bool) -> list[LocalEvent]:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if not len(samples):
            return []
        start = self.clock
        self.clock += len(samples)
        events: list[LocalEvent] = []

        if speech and not self.voiced:
            self.activity_epoch += 1
            self.floor_sent = False
            self.has_spoken = True
            events.append(SpeechStarted(self.activity_epoch))
            if not self.turn_parts:
                self.turn_start = max(0, start - self.preroll_length)
                for part in self.preroll:
                    self._append(part)
        self.voiced = speech
        if speech:
            self.silence_samples = 0
            self.last_check = 0
        else:
            self.silence_samples += len(samples)
        if self.turn_parts or speech:
            self._append(samples)

        if self.turn_parts and self.turn_length >= self.max_samples:
            # Um turno longo é um chunk parcial, NUNCA uma conclusão artificial.
            events.append(self._finish(complete=False, end_sample=self.clock))
            if self.voiced:
                self.turn_start = self.clock
            self.silence_samples = 0

        if self.turn_parts and not speech and self.silence_samples >= self.min_pause:
            check = (self.last_check == 0 or
                     self.silence_samples - self.last_check >= self.recheck)
            completed = False
            if check:
                self.last_check = self.silence_samples
                recent = self._audio()[-8 * SAMPLE_RATE:]
                completed = self.classifier.complete(recent)
            if completed or self.silence_samples >= self.max_pause:
                end_sample = self.clock - self.silence_samples
                events.append(self._finish(complete=True, end_sample=end_sample))

        # Só sinaliza abertura depois de ao menos uma fala e sem turno pendente.
        if (not speech and not self.turn_parts and self.has_spoken
                and not self.floor_sent and self.silence_samples >= self.floor_silence):
            self.floor_sent = True
            events.append(FloorOpened(self.activity_epoch,
                                      self.silence_samples * 1000 // SAMPLE_RATE))

        self._remember_preroll(samples)
        return events

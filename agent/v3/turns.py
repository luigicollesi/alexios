"""Detecção de turnos baseada em VAD + decisão semântica (injetáveis/testáveis).

O áudio é acumulado apenas localmente. Fim de turno NÃO autoriza falar.
Uma abertura de fala só é emitida durante silêncio contínuo após um turno.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol
import time
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
    # Somente amostras marcadas como fala pelo VAD; não muda o áudio do Whisper.
    speaker_audio: np.ndarray | None = None


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
        diagnostic: Callable[[str], None] | None = None,
    ) -> None:
        if not (0 < min_pause_ms < max_pause_ms):
            raise ValueError("Intervalos de pausa inválidos.")
        if floor_silence_ms < min_pause_ms or recheck_ms <= 0 or max_segment_s < 5:
            raise ValueError("Limites da detecção inválidos.")
        self.classifier = classifier
        self.diagnostic = diagnostic
        self.min_pause = min_pause_ms * SAMPLE_RATE // 1000
        self.recheck = recheck_ms * SAMPLE_RATE // 1000
        self.max_pause = max_pause_ms * SAMPLE_RATE // 1000
        self.floor_silence = floor_silence_ms * SAMPLE_RATE // 1000
        self.max_samples = max_segment_s * SAMPLE_RATE
        # Convert milliseconds to samples. Without // 1000, 192 ms became
        # 192 seconds of replayed audio across consecutive speech turns.
        self.preroll_samples = preroll_ms * SAMPLE_RATE // 1000
        self.clock = 0
        self.activity_epoch = initial_epoch
        self.has_spoken = False
        self.floor_sent = False
        self.voiced = False
        self.silence_samples = 0
        self.last_check = 0
        self.turn_start = 0
        # Sample-accurate bounds for the current utterance, independent of
        # stale turn_parts: an old buffer must never enter a new ASR request.
        self.first_voice_sample: int | None = None
        self.last_voice_sample: int | None = None
        # Indices in turn_parts, not inferred from stale absolute timestamps.
        self.first_voice_offset: int | None = None
        self.last_voice_offset: int | None = None
        self.turn_parts: list[np.ndarray] = []
        self.turn_length = 0
        self.voice_parts: deque[np.ndarray] = deque()
        self.voice_length = 0
        self.max_voice_samples = 8 * SAMPLE_RATE
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

    def _append_voice(self, frame: np.ndarray) -> None:
        """Conserva os últimos oito segundos de fala efetiva sem o silêncio."""
        self.voice_parts.append(frame.copy())
        self.voice_length += len(frame)
        while self.voice_parts and self.voice_length > self.max_voice_samples:
            old = self.voice_parts.popleft()
            self.voice_length -= len(old)

    def _voice_audio(self) -> np.ndarray:
        if not self.voice_parts:
            return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(self.voice_parts), dtype=np.float32)

    def _finish(self, *, complete: bool, end_sample: int) -> CapturedTurn:
        raw_audio = self._audio()
        first = self.first_voice_sample
        last = self.last_voice_sample
        if (first is not None and last is not None
                and self.first_voice_offset is not None
                and self.last_voice_offset is not None):
            # Indices are tracked as new frames are appended, so an
            # accidentally retained prefix cannot shift the selected voice.
            left = max(0, self.first_voice_offset - self.preroll_samples)
            right = min(len(raw_audio), self.last_voice_offset + self.min_pause)
            audio = np.ascontiguousarray(raw_audio[left:right])
            # Absolute metadata reflects new voiced frames, not a stale
            # zero-based buffer. Preserve a small pre-roll for Whisper.
            start_sample = max(0, first - min(self.preroll_samples,
                                               self.first_voice_offset))
        else:
            audio = np.ascontiguousarray(raw_audio)
            start_sample = max(0, self.clock - len(raw_audio))
        turn = CapturedTurn(
            turn_id=str(uuid4()),
            audio=audio,
            started_ms=start_sample * 1000 // SAMPLE_RATE,
            ended_ms=end_sample * 1000 // SAMPLE_RATE,
            complete=complete,
            activity_epoch=self.activity_epoch,
            speaker_audio=self._voice_audio(),
        )
        # Drop *all* audio references as soon as a CapturedTurn is emitted.
        self.voice_parts.clear()
        self.voice_length = 0
        self.turn_parts.clear()
        self.turn_length = 0
        self.first_voice_sample = None
        self.last_voice_sample = None
        self.first_voice_offset = None
        self.last_voice_offset = None
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
            # A completed turn has no active voiced span. If any old
            # turn_parts survived, discard them before starting a new turn.
            # A genuine unfinished pause still has first_voice_sample set.
            if self.first_voice_sample is None and self.turn_parts:
                self.turn_parts.clear()
                self.turn_length = 0
                self.voice_parts.clear()
                self.voice_length = 0
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
            if self.first_voice_sample is None:
                self.first_voice_sample = start
                self.first_voice_offset = self.turn_length
            self.last_voice_sample = self.clock
            self.last_voice_offset = self.turn_length + len(samples)
            self.silence_samples = 0
            self.last_check = 0
            self._append_voice(samples)
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
                started = time.monotonic()
                completed = self.classifier.complete(recent)
                if self.diagnostic:
                    elapsed_ms = round((time.monotonic() - started) * 1000)
                    pause_ms = self.silence_samples * 1000 // SAMPLE_RATE
                    self.diagnostic(
                        f"pausa={pause_ms}ms, turno_completo={completed}, "
                        f"inferencia={elapsed_ms}ms"
                    )
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

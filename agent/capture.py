"""Captura: callback leve, VAD/Smart Turn fora da thread de áudio.

A thread nativa não executa inferência, rede nem Whisper. O buffer é limitado
e falhas de sobrecarga param a gravação (nunca descartam fala silenciosamente).
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

from .dispatch import EventDispatcher
from .models import SileroVad
from .turns import CapturedTurn, FloorOpened, SpeechStarted, TurnEngine


class TurnAwareCapture:
    def __init__(
        self, dispatcher: EventDispatcher, vad: SileroVad,
        classifier: object, device: int | None, *, debug_audio: bool = False,
    ) -> None:
        self.dispatcher = dispatcher
        self.vad = vad
        self.classifier = classifier
        self.device = device
        self.debug_audio = debug_audio
        self.active = threading.Event()
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.last_error: str | None = None
        # 16 segundos de folga se Smart Turn/Whisper estiverem sob carga em ARM.
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=512)
        self.thread: threading.Thread | None = None
        self._stream: Any | None = None

    @property
    def running(self) -> bool:
        return self.active.is_set()

    def _audio_callback(self, indata, frames: int, _time_info, status) -> None:
        if status:
            print(f"Aviso de áudio: {status}", file=sys.stderr)
        if not self.active.is_set():
            return
        if frames != self.vad.window_size:
            self.last_error = (
                f"Dispositivo entregou {frames} amostras; "
                f"esperadas {self.vad.window_size}."
            )
            self.active.clear()
            return
        try:
            self.frames.put_nowait(np.asarray(indata[:, 0], dtype=np.float32).copy())
        except queue.Full:
            self.last_error = "Fila de áudio saturada; captura suspensa com segurança."
            self.active.clear()

    def _clear_frames(self) -> None:
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return

    def start(self) -> bool:
        if self.closed.is_set():
            return False
        if self.active.is_set():
            return True
        if self.dispatcher.failure:
            self.last_error = self.dispatcher.failure
            return False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.last_error = "Thread anterior do microfone não finalizou."
                return False
        self._clear_frames()
        self.vad.reset()
        self.last_error = None
        self.ready.clear()
        self.active.set()
        self.thread = threading.Thread(
            target=self._record_loop, name="alexios-v3-capture", daemon=True,
        )
        self.thread.start()
        if not self.ready.wait(timeout=8):
            self.last_error = "Tempo excedido na abertura do microfone."
            self.stop()
            return False
        if not self.active.is_set() or self.last_error:
            self.stop()
            return False
        return True

    def _process_loop(self) -> None:
        engine = TurnEngine(
            self.classifier,
            initial_epoch=self.dispatcher.activity_epoch,
            diagnostic=(lambda msg: print(f"[SMART TURN] {msg}", flush=True))
                       if self.debug_audio else None,
        )
        last_audio_log = time.monotonic()
        accumulated_energy = 0.0
        accumulated_samples = 0
        peak = 0.0
        last_detected = False
        try:
            while self.active.is_set() and not self.closed.is_set():
                if self.dispatcher.failure:
                    raise RuntimeError(self.dispatcher.failure)
                try:
                    samples = self.frames.get(timeout=0.2)
                except queue.Empty:
                    continue
                speech = self.vad.is_speech(samples)
                if self.debug_audio:
                    accumulated_energy += float(np.dot(samples, samples))
                    accumulated_samples += len(samples)
                    peak = max(peak, float(np.max(np.abs(samples))))
                    now = time.monotonic()
                    if now - last_audio_log >= 2:
                        rms = (accumulated_energy / max(1, accumulated_samples)) ** 0.5
                        print(
                            f"[AUDIO] rms={rms:.4f} pico={peak:.4f} "
                            f"vad={'FALA' if speech else 'silencio'} "
                            f"fila={self.frames.qsize()}/{self.frames.maxsize}",
                            flush=True,
                        )
                        if peak < 0.003:
                            print(
                                "[AUDIO] sinal extremamente baixo: confira --device, "
                                "microfone e volume de entrada.",
                                flush=True,
                            )
                        last_audio_log = now
                        accumulated_energy = 0.0
                        accumulated_samples = 0
                        peak = 0.0
                if speech != last_detected:
                    if self.debug_audio:
                        print(
                            f"[SILERO] {'fala detectada' if speech else 'pausa detectada'}",
                            flush=True,
                        )
                    last_detected = speech
                events = engine.feed(samples, speech)
                for event in events:
                    if isinstance(event, SpeechStarted):
                        print(f"[V3] speech_started epoch={event.activity_epoch}", flush=True)
                        self.dispatcher.speech_started(event.activity_epoch)
                    elif isinstance(event, (CapturedTurn, FloorOpened)):
                        if isinstance(event, CapturedTurn):
                            print(
                                f"[V3] {'turn_end' if event.complete else 'turn_chunk'} "
                                f"duracao={len(event.audio)/16000:.1f}s "
                                f"epoch={event.activity_epoch} -> Whisper",
                                flush=True,
                            )
                        else:
                            print(
                                f"[V3] floor_open candidato "
                                f"epoch={event.activity_epoch} silencio={event.silence_ms}ms",
                                flush=True,
                            )
                        self.dispatcher.enqueue(event)
        finally:
            # Em FIM, uma fala parcial não deve desaparecer silenciosamente;
            # ela é um chunk, nunca um turn_end que autorizaria a fala.
            if not self.dispatcher.failure:
                fragment = engine.flush_partial()
                if fragment is not None:
                    try:
                        self.dispatcher.enqueue(fragment)
                    except RuntimeError as exc:
                        self.last_error = str(exc)

    def _record_loop(self) -> None:
        try:
            with sd.InputStream(
                samplerate=16_000,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=self.vad.window_size,
                callback=self._audio_callback,
            ):
                self.ready.set()
                self._process_loop()
        except Exception as exc:
            self.last_error = str(exc)
            print(f"Captura V3 interrompida: {exc}", file=sys.stderr)
        finally:
            self.active.clear()
            self.ready.set()
            if self.last_error:
                print(f"[V3] Microfone interrompido: {self.last_error}", file=sys.stderr, flush=True)

    def stop(self) -> None:
        self.active.clear()
        self.dispatcher.cancel_floor()
        if self.thread and self.thread.is_alive() and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
        self._clear_frames()

    def shutdown(self) -> None:
        self.stop()
        self.closed.set()
        self.dispatcher.shutdown()

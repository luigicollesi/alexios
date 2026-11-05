"""Captura 4s/3s, transcrição Whisper e envio somente de palavras ao servidor V2."""

from __future__ import annotations

import queue
import sys
import threading
import time
from uuid import uuid4

import numpy as np
import sounddevice as sd

from audio_windows import SAMPLE_RATE, WindowCutter
from .bridge import BridgeClient, BridgeError

FRAME_SAMPLES = SAMPLE_RATE // 4
MAX_AUDIO_QUEUE = 8


class AudioCapture:
    def __init__(self, client: BridgeClient, model: object, device: int | None) -> None:
        self.client = client
        self.model = model
        self.device = device
        self.active = threading.Event()
        self.exiting = threading.Event()
        self.ready = threading.Event()
        self.queue: queue.Queue[tuple[str, int, int, int, np.ndarray]] = queue.Queue(
            maxsize=MAX_AUDIO_QUEUE,
        )
        self.lock = threading.Lock()
        self.generation = 0
        self.capture_id = ""
        self.last_error: str | None = None
        self.recording: threading.Thread | None = None
        self.transcriber = threading.Thread(
            target=self._transcribe_loop, name="alexios-v2-stt", daemon=True,
        )
        self.transcriber.start()

    @property
    def running(self) -> bool:
        return self.active.is_set()

    def _current(self, generation: int, capture_id: str) -> bool:
        with self.lock:
            return (
                self.active.is_set()
                and not self.exiting.is_set()
                and self.generation == generation
                and self.capture_id == capture_id
            )

    def _empty_queue(self) -> None:
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def start(self) -> bool:
        if self.active.is_set():
            return True
        if self.exiting.is_set():
            return False
        if self.recording and self.recording.is_alive():
            self.recording.join(timeout=3)
            if self.recording.is_alive():
                self.last_error = "A captura anterior ainda está encerrando."
                return False
        self._empty_queue()
        self.ready.clear()
        self.last_error = None
        with self.lock:
            self.generation += 1
            generation = self.generation
            capture_id = str(uuid4())
            self.capture_id = capture_id
            self.active.set()
        self.recording = threading.Thread(
            target=self._record_loop, args=(generation, capture_id),
            name="alexios-v2-microphone", daemon=True,
        )
        self.recording.start()
        if not self.ready.wait(timeout=8):
            self.last_error = "Tempo esgotado ao abrir o microfone."
            self.stop()
            return False
        if not self.active.is_set() or self.last_error:
            self.stop()
            return False
        return True

    def stop(self) -> None:
        self.active.clear()
        with self.lock:
            self.generation += 1
            self.capture_id = ""
        self._empty_queue()
        if self.recording and self.recording.is_alive() and self.recording is not threading.current_thread():
            self.recording.join(timeout=3)

    def shutdown(self) -> None:
        self.stop()
        self.exiting.set()
        # Workers are daemonized because native STT calls cannot be interrupted safely.
        # They stop cooperatively after returning from their bounded network/STT work.
        if self.transcriber.is_alive():
            self.transcriber.join(timeout=2)

    def _record_loop(self, generation: int, capture_id: str) -> None:
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                device=self.device, blocksize=FRAME_SAMPLES,
            ) as stream:
                cutter = WindowCutter()
                self.ready.set()
                while self._current(generation, capture_id):
                    samples, overflow = stream.read(FRAME_SAMPLES)
                    if overflow:
                        print("Aviso: overflow do microfone.", file=sys.stderr)
                    for window in cutter.push(samples[:, 0]):
                        if not self._current(generation, capture_id):
                            break
                        item = (capture_id, generation, window.index, window.start_ms, window.samples)
                        try:
                            self.queue.put_nowait(item)
                        except queue.Full:
                            # A fila permanece limitada; inicia-se uma nova sequência coerente.
                            print("Whisper atrasado; reiniciando captura de janelas.", file=sys.stderr)
                            self._empty_queue()
                            cutter = WindowCutter()
                            capture_id = str(uuid4())
                            with self.lock:
                                if self.generation != generation or not self.active.is_set():
                                    break
                                self.capture_id = capture_id
                            break
        except (sd.PortAudioError, ValueError, OSError) as exc:
            self.last_error = str(exc)
            print(f"Microfone indisponível: {exc}", file=sys.stderr)
        finally:
            self.ready.set()
            with self.lock:
                if self.generation == generation:
                    self.active.clear()

    def _transcribe_loop(self) -> None:
        while not self.exiting.is_set():
            try:
                capture_id, generation, index, start_ms, samples = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if not self._current(generation, capture_id):
                continue
            words: list[dict] = []
            if samples.size > 0 and float(np.max(np.abs(samples))) >= 0.002:
                try:
                    segments, _ = self.model.transcribe(
                        samples, language="pt", beam_size=3, vad_filter=True,
                        word_timestamps=True, condition_on_previous_text=False,
                    )
                    for segment in segments:
                        for word in segment.words or []:
                            content = word.word.strip()
                            if not content:
                                continue
                            start = start_ms + max(0, round(word.start * 1000))
                            end = max(start, start_ms + max(0, round(word.end * 1000)))
                            words.append({"text": content, "startMs": start, "endMs": end})
                except Exception as exc:
                    print(f"Falha de transcrição: {exc}", file=sys.stderr)
                    # Não deixa uma lacuna na sequência: envia a janela vazia.
                    words = []
            if not self._current(generation, capture_id):
                continue
            delay = 1
            while self._current(generation, capture_id):
                try:
                    accepted = self.client.upload(capture_id, index, words)
                    if accepted and words:
                        print(f"[{start_ms / 1000:.0f}s] " + " ".join(w["text"] for w in words))
                    break
                except BridgeError as exc:
                    if exc.fatal or exc.conflict:
                        print(f"Janela {index} não aceita: {exc}", file=sys.stderr)
                        break
                    print(f"Falha no envio da janela {index}; repetindo: {exc}", file=sys.stderr)
                    time.sleep(delay)
                    delay = min(delay * 2, 8)

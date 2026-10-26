"""Continuous ambient-listening mode for Alexios.

Runs alongside the existing interactive main.py, without a wake word.
Audio is processed locally; only time-stamped recognized words leave this computer.
"""

import argparse
import base64
import json
import os
from pathlib import Path
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from uuid import uuid4

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from audio_windows import SAMPLE_RATE, WindowCutter

FRAME_SAMPLES = SAMPLE_RATE // 4
NETWORK_TIMEOUT = 35
OFFLINE_PAUSE_AFTER = 35


class BridgeClient:
    def __init__(self, url: str, token: str, instance_id: str) -> None:
        base = urllib.parse.urlsplit(url.rstrip("/"))
        if base.scheme != "https" and not (
            base.scheme == "http" and base.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("ALEXIOS_BRIDGE_URL exige HTTPS (HTTP apenas local)")
        if not base.netloc or not token:
            raise ValueError("Configure ALEXIOS_BRIDGE_URL e ALEXIOS_AGENT_TOKEN")
        self.base = url.rstrip("/")
        self.token = token
        self.instance_id = instance_id

    def request(self, method: str, path: str, body: dict | None = None, timeout: int = NETWORK_TIMEOUT) -> dict:
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=payload,
            method=method,
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/json",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Bridge HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Bridge indisponível: {exc}") from exc

    def poll(self, command_seq: int, speech_seq: int) -> dict:
        query = urllib.parse.urlencode({
            "instanceId": self.instance_id,
            "after": command_seq,
            "speechAfter": speech_seq,
        })
        return self.request("GET", "/v1/agent/poll?" + query)

    def report(self, power: str) -> None:
        self.request("POST", "/v1/agent/state", {
            "instanceId": self.instance_id, "power": power,
        })

    def upload(self, capture_id: str, index: int, words: list[dict]) -> None:
        self.request("POST", "/v1/transcript", {
            "instanceId": self.instance_id, "captureId": capture_id,
            "windowIndex": index, "words": words,
        })

    def analyze(self) -> dict:
        """Independent 8-second context cycle on the Vercel API."""
        return self.request("POST", "/v1/agent/analyze", {
            "instanceId": self.instance_id,
        }, timeout=110)


class LocalSpeaker:
    """Interruptible system speech. Skips recording while the local TTS is audible."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.is_speaking = threading.Event()
        self.messages: queue.Queue[str] = queue.Queue(maxsize=4)
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._generation = 0
        threading.Thread(target=self._run, name="alexios-tts", daemon=True).start()

    def say(self, text: str) -> None:
        if not self.enabled or not text.strip():
            return
        try:
            self.messages.put_nowait(text[:600])
        except queue.Full:
            print("TTS sobrecarregado; fala descartada.", file=sys.stderr)

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            if self._process and self._process.poll() is None:
                self._process.terminate()
            while not self.messages.empty():
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    break

    def _command(self, text: str) -> tuple[list[str], bytes | None] | None:
        system = platform.system()
        if system == "Windows":
            encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
            script = (
                "Add-Type -AssemblyName System.Speech;"
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.Speak([Text.Encoding]::UTF8.GetString("
                f"[Convert]::FromBase64String('{encoded_text}')));"
            )
            return ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand",
                    base64.b64encode(script.encode("utf-16-le")).decode("ascii")], None
        if system == "Darwin" and shutil.which("say"):
            return ["say", text], None
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            return [executable, "--stdin", "-v", "pt-br"], text.encode("utf-8")
        return None

    def _run(self) -> None:
        warned = False
        while True:
            text = self.messages.get()
            command = self._command(text)
            if command is None:
                if not warned:
                    print("TTS do sistema ausente. Instale espeak-ng ou use --no-tts.", file=sys.stderr)
                    warned = True
                continue
            arguments, input_bytes = command
            with self._lock:
                generation = self._generation
                self.is_speaking.set()
                try:
                    self._process = subprocess.Popen(
                        arguments, stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                except OSError as exc:
                    print(f"TTS indisponível: {exc}", file=sys.stderr)
                    self._process = None
                    self.is_speaking.clear()
                    continue
            try:
                self._process.communicate(input=input_bytes)
            finally:
                with self._lock:
                    if self._generation == generation:
                        self.is_speaking.clear()
                    self._process = None
                    self.is_speaking.clear()


class AmbientRuntime:
    def __init__(self, client: BridgeClient, model: WhisperModel,
                 device: int | None, speaker: LocalSpeaker) -> None:
        self.client = client
        self.model = model
        self.device = device
        self.speaker = speaker
        self.active = threading.Event()
        self.started = threading.Event()
        self.capture_id = ""
        self.queue: queue.Queue[tuple[str, int, int, np.ndarray]] = queue.Queue(maxsize=8)
        self.lock = threading.Lock()
        self.recording: threading.Thread | None = None
        self.analysis_thread: threading.Thread | None = None
        self.generation = 0
        self.state = "stopped"
        threading.Thread(target=self._transcribe_loop, name="alexios-stt", daemon=True).start()

    def report(self, power: str) -> None:
        self.state = power
        try:
            self.client.report(power)
        except RuntimeError as exc:
            print(f"Estado não confirmado: {exc}", file=sys.stderr)

    def start(self) -> None:
        if self.active.is_set():
            return
        self.report("starting")
        self.started.clear()
        with self.lock:
            self.generation += 1
            generation = self.generation
        self.active.set()
        self.recording = threading.Thread(target=self._record_loop, name="alexios-mic", daemon=True)
        self.recording.start()
        if self.started.wait(timeout=8) and self.active.is_set():
            self.report("running")
            self.analysis_thread = threading.Thread(
                target=self._analysis_loop, args=(generation,),
                name="alexios-context", daemon=True,
            )
            self.analysis_thread.start()
            print("ESCUTA ATIVA: janelas de 4s, deslocamento de 3s; análise a cada 8s.")
        else:
            self.active.clear()
            self.report("stopped")
            print("Não foi possível iniciar o microfone.", file=sys.stderr)

    def stop(self, report: bool = True) -> None:
        self.active.clear()
        self.speaker.stop()
        with self.lock:
            self.generation += 1
            self.capture_id = ""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        if self.recording and self.recording.is_alive():
            self.recording.join(timeout=2)
        if report:
            self.report("stopped")
        else:
            self.state = "stopped"
        print("Escuta desligada; long polling permanece ativo.")

    def _record_loop(self) -> None:
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                device=self.device, blocksize=FRAME_SAMPLES) as stream:
                cutter = WindowCutter()
                capture = str(uuid4())
                with self.lock:
                    self.capture_id = capture
                was_muted = False
                self.started.set()
                while self.active.is_set():
                    samples, overflow = stream.read(FRAME_SAMPLES)
                    if overflow:
                        print("Aviso: overflow do microfone; trecho pode estar incompleto.",
                              file=sys.stderr)
                    muted = self.speaker.is_speaking.is_set()
                    if muted:
                        was_muted = True
                        continue
                    if was_muted:
                        # Echo protection: do not transcribe Alexios's own voice.
                        cutter = WindowCutter()
                        capture = str(uuid4())
                        with self.lock:
                            self.capture_id = capture
                        was_muted = False
                    for window in cutter.push(samples[:, 0]):
                        item = (capture, window.index, window.start_ms, window.samples)
                        try:
                            self.queue.put_nowait(item)
                        except queue.Full:
                            # Fail closed rather than accumulate an unbounded STT backlog.
                            print("STT atrasado; reiniciando a sequência de janelas.",
                                  file=sys.stderr)
                            while not self.queue.empty():
                                try:
                                    self.queue.get_nowait()
                                except queue.Empty:
                                    break
                            cutter = WindowCutter()
                            capture = str(uuid4())
                            with self.lock:
                                self.capture_id = capture
                            break
        except (sd.PortAudioError, ValueError) as exc:
            print(f"Falha do microfone: {exc}", file=sys.stderr)
            self.active.clear()
            self.started.set()
            self.report("stopped")

    def _analysis_loop(self, generation: int) -> None:
        """A timer owned by the local Agent triggers one serverless request every 8s."""
        next_cycle = time.monotonic() + 8.0
        while self.active.is_set() and self.generation == generation:
            while self.active.is_set() and self.generation == generation:
                remaining = next_cycle - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(remaining, 0.25))
            if not self.active.is_set() or self.generation != generation:
                return
            next_cycle = time.monotonic() + 8.0
            try:
                response = self.client.analyze()
                if response.get("accepted"):
                    print("Contexto atualizado; postura:", response.get("pose"))
            except RuntimeError as exc:
                print(f"Falha no ciclo contextual: {exc}", file=sys.stderr)

    def _transcribe_loop(self) -> None:
        while True:
            capture, index, start_ms, samples = self.queue.get()
            if not self.active.is_set() or capture != self.capture_id:
                continue
            if np.max(np.abs(samples)) < 0.002:
                words = []
            else:
                try:
                    segments, _ = self.model.transcribe(
                        samples, language="pt", beam_size=3, vad_filter=True,
                        word_timestamps=True, condition_on_previous_text=False,
                    )
                    words = []
                    for segment in segments:
                        for word in segment.words or []:
                            text = word.word.strip()
                            if text:
                                words.append({
                                    "text": text,
                                    "startMs": start_ms + max(0, round(word.start * 1000)),
                                    "endMs": start_ms + max(0, round(word.end * 1000)),
                                })
                except Exception as exc:
                    print(f"Falha STT: {exc}", file=sys.stderr)
                    continue
            if not self.active.is_set() or capture != self.capture_id:
                continue
            try:
                self.client.upload(capture, index, words)
                if words:
                    print(f"[{start_ms / 1000:.0f}s] " + " ".join(w["text"] for w in words))
            except RuntimeError as exc:
                print(f"Transcrição não enviada: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Alexios: escuta contínua sem wake word")
    parser.add_argument("--device", type=int, default=None, help="Índice do microfone")
    parser.add_argument("--model", default="base", help="Modelo Faster-Whisper local")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--no-tts", action="store_true", help="Texto no dashboard sem áudio de saída")
    args = parser.parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    load_dotenv(Path(__file__).resolve().parent / ".env")
    url = os.getenv("ALEXIOS_BRIDGE_URL", "https://alexios-web.vercel.app")
    token = os.getenv("ALEXIOS_AGENT_TOKEN", "")
    instance_id = os.getenv("ALEXIOS_INSTANCE_ID") or "alexios-" + uuid4().hex[:16]
    client = BridgeClient(url, token, instance_id)
    print(f"Whisper local: {args.model} (CPU/int8)")
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    runtime = AmbientRuntime(client, model, args.device, LocalSpeaker(not args.no_tts))

    command_seq = -1
    speech_seq = -1
    retry_delay = 1
    last_success = time.monotonic()
    try:
        while True:
            try:
                response = client.poll(command_seq, speech_seq)
                last_success = time.monotonic()
                retry_delay = 1
            except RuntimeError as exc:
                print(f"Long polling: {exc}", file=sys.stderr)
                if runtime.active.is_set() and time.monotonic() - last_success > OFFLINE_PAUSE_AFTER:
                    runtime.stop(report=False)
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 10)
                continue
            command_seq = int(response["commandSeq"])
            incoming_speech_seq = int(response["speechSeq"])
            desired = response["desiredPower"]
            if desired == "stopped" and runtime.active.is_set():
                runtime.stop()
            elif desired == "running" and not runtime.active.is_set():
                runtime.start()

            speech = response.get("speech")
            if (speech_seq >= 0 and incoming_speech_seq > speech_seq and
                    desired == "running" and runtime.active.is_set() and isinstance(speech, dict)):
                text = str(speech.get("text", ""))
                if text:
                    print("Alexios:", text)
                    runtime.speaker.say(text)
            speech_seq = incoming_speech_seq
    except KeyboardInterrupt:
        print("\nEncerrando Agent.")
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()

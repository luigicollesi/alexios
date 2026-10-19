"""Alexios: microfone -> transcrição local -> OpenRouter -> resposta em texto."""

import argparse
import queue
import sys
from collections import deque
import urllib.error
import urllib.request
import json
import os
from pathlib import Path

import numpy as np
from openwakeword.model import Model as WakeWordModel
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from context import BASE_CONTEXT


SAMPLE_RATE = 16000
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
MAX_HISTORY_MESSAGES = 12


def record_audio(device: int | None, engine, threshold: float) -> np.ndarray:
    """Espera pela palavra-chave e grava até silêncio, sem usar Enter."""
    frames: queue.Queue[bytes] = queue.Queue(maxsize=128)
    frame_length = 1280  # 80 ms de áudio mono a 16 kHz
    frame_seconds = frame_length / SAMPLE_RATE
    preroll = deque(maxlen=max(1, round(0.4 / frame_seconds)))
    silence_limit = max(1, round(1.1 / frame_seconds))
    speech_timeout = max(1, round(3.5 / frame_seconds))
    max_frames = round(20 / frame_seconds)

    def callback(indata, count, time_info, status):
        if status:
            print(f"Aviso do microfone: {status}", file=sys.stderr)
        try:
            frames.put_nowait(bytes(indata))
        except queue.Full:
            pass

    print("Aguardando 'Aléxios'... (Ctrl+C para sair)")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=frame_length,
        device=device, channels=1, dtype="int16", callback=callback,
    ):
        # O fluxo fica aberto durante a escuta; fecha antes da chamada à API.
        while True:
            raw = frames.get()
            pcm = np.frombuffer(raw, dtype=np.int16)
            scores = engine.predict(pcm)
            if any(score >= 0.5 for score in scores.values()):
                engine.reset()
                print("Aléxios detectado. Ouvindo comando...")
                captured = list(preroll) + [raw]
                heard_speech = False
                silence = 0
                wait_frames = 0
                break
            preroll.append(raw)

        while len(captured) < max_frames:
            raw = frames.get()
            captured.append(raw)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples * samples)))
            if rms >= threshold:
                heard_speech = True
                silence = 0
            elif heard_speech:
                silence += 1
                if silence >= silence_limit:
                    break
            else:
                wait_frames += 1
                if wait_frames >= speech_timeout:
                    break

    if not heard_speech:
        print("Comando não detectado após ativação.")
        return np.empty(0, dtype=np.float32)
    return np.frombuffer(b"".join(captured), dtype=np.int16).astype(np.float32) / 32768.0


def create_wake_engine(root: Path) -> WakeWordModel:
    """Carrega um modelo ONNX personalizado de 'Aléxios', sem conta/API."""
    keyword = root / "models" / "alexios.onnx"
    if not keyword.is_file():
        raise RuntimeError(
            f"Modelo de ativação ausente: {keyword}. "
            "É necessário treinar/obter um modelo openWakeWord personalizado "
            "para 'Aléxios' em formato .onnx (consulte README)."
        )
    return WakeWordModel(
        wakeword_models=[str(keyword)], inference_framework="onnx"
    )


def transcribe_audio(model: WhisperModel, audio: np.ndarray) -> str:
    """Transcreve a fala localmente para português."""
    segments, _ = model.transcribe(
        audio, language="pt", beam_size=5, vad_filter=True
    )
    return " ".join(
        segment.text.strip() for segment in segments if segment.text.strip()
    )


def ask_openrouter(api_key: str, messages: list[dict[str, str]]) -> str:
    """Envia somente texto ao modelo e retorna o texto da resposta."""
    payload = json.dumps({
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": BASE_CONTEXT}, *messages],
        "max_tokens": 800,
        "stream": False,
    }).encode("utf-8")

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {details}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Falha de conexão com OpenRouter: {exc}") from exc

    try:
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("O modelo não retornou texto (resposta vazia).")
        return content.strip()
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Resposta inesperada da API: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assistente Alexios: microfone, Whisper e OpenRouter"
    )
    parser.add_argument("--list-devices", action="store_true", help="Lista microfones disponíveis")
    parser.add_argument("--device", type=int, default=None, help="ID do microfone")
    parser.add_argument("--model", default="base", help="Modelo Whisper (tiny, base, small)")
    parser.add_argument("--threshold", type=float, default=0.018, help="Limiar RMS de detecção de fala")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("OPENROUTER_API_KEY não encontrada. Crie o arquivo .env na raiz.", file=sys.stderr)
        raise SystemExit(1)

    try:
        engine = create_wake_engine(Path(__file__).resolve().parent)
    except Exception as exc:
        print(f"Falha ao inicializar detector de voz: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    try:
        info = sd.query_devices(args.device, "input")
        sd.check_input_settings(
            device=args.device, samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        print(f"Microfone: {info['name']}")
        print(f"Carregando modelo Whisper '{args.model}' na CPU...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        history: list[dict[str, str]] = []

        while True:
            audio = record_audio(args.device, engine, args.threshold)
            if audio.size == 0:
                print("Nenhuma fala detectada. Tente novamente.")
            else:
                print("Transcrevendo...")
                transcription = transcribe_audio(model, audio)
                if not transcription:
                    print("Nenhuma fala reconhecida.")
                else:
                    print(f"Você: {transcription}")
                    print("Consultando a IA...")
                    candidate = [*history, {"role": "user", "content": transcription}]
                    try:
                        answer = ask_openrouter(api_key, candidate)
                    except RuntimeError as exc:
                        print(f"Erro ao consultar a IA: {exc}", file=sys.stderr)
                    else:
                        print(f"Alexios: {answer}\n")
                        history = [
                            *candidate, {"role": "assistant", "content": answer}
                        ][-MAX_HISTORY_MESSAGES:]

    except (sd.PortAudioError, ValueError) as exc:
        print(f"Erro ao acessar o microfone: {exc}", file=sys.stderr)
        print("Use 'python main.py --list-devices' para selecionar outro dispositivo.")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nEncerrando.")


if __name__ == "__main__":
    main()

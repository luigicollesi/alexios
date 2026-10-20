"""Alexios: microfone -> transcrição local -> OpenRouter -> resposta em texto."""

import argparse
import queue
import sys
import urllib.error
import urllib.request
import json
import os
from pathlib import Path

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from context import BASE_CONTEXT


SAMPLE_RATE = 16000
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
MAX_HISTORY_MESSAGES = 12


def record_audio(device: int | None, max_seconds: int = 60) -> np.ndarray:
    """Grava áudio mono enquanto o usuário não pressiona Enter."""
    chunks: queue.Queue[np.ndarray] = queue.Queue()
    errors: list[str] = []

    def callback(indata, frames, time_info, status):
        if status:
            errors.append(str(status))
        chunks.put(indata.copy())

    input("Pressione Enter para INICIAR a gravação...")
    print("Gravando. Fale agora e pressione Enter para PARAR.")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        callback=callback,
    ):
        input()

    if errors:
        print(f"Aviso do microfone: {errors[-1]}", file=sys.stderr)

    recorded = []
    while not chunks.empty():
        recorded.append(chunks.get_nowait())

    if not recorded:
        return np.empty(0, dtype=np.float32)

    audio = np.concatenate(recorded, axis=0).reshape(-1)
    if audio.size > max_seconds * SAMPLE_RATE:
        print(f"Áudio limitado aos primeiros {max_seconds} segundos.")
        audio = audio[: max_seconds * SAMPLE_RATE]
    return audio


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
        info = sd.query_devices(args.device, "input")
        sd.check_input_settings(
            device=args.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32"
        )
        print(f"Microfone: {info['name']}")
        print(f"Carregando modelo Whisper '{args.model}' na CPU...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
        history: list[dict[str, str]] = []

        while True:
            audio = record_audio(args.device)
            if audio.size == 0 or np.max(np.abs(audio)) < 0.001:
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

            if input("Nova gravação? [Enter = sim / n = sair]: ").strip().lower() in {
                "n", "não", "nao"
            }:
                break
    except (sd.PortAudioError, ValueError) as exc:
        print(f"Erro ao acessar o microfone: {exc}", file=sys.stderr)
        print("Use 'python main.py --list-devices' para selecionar outro dispositivo.")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nEncerrando.")


if __name__ == "__main__":
    main()

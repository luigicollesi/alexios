"""Alexios V2: microfone e Whisper locais; IA e personagem exclusivos na web.

Execute: python continuous.py --model base
Ctrl+C / SIGTERM encerra o microfone e o long polling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

import sounddevice as sd
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from agent.bridge import BridgeClient, BridgeError
from agent.capture import AudioCapture
from agent.config import AgentConfig
from agent.runtime import AgentRuntime


def shutdown_signal(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Alexios V2: captura local, IA na Vercel")
    parser.add_argument("--device", type=int, default=None, help="Índice do microfone")
    parser.add_argument("--model", default="base", help="Modelo Faster-Whisper (tiny, base, small)")
    parser.add_argument("--list-devices", action="store_true", help="Listar dispositivos de áudio")
    parser.add_argument(
        "--no-tts", action="store_true",
        help="Compatibilidade com CLI anterior; o modo V2 nunca executa TTS local",
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        config = AgentConfig.from_env()
    except ValueError as exc:
        print(f"Configuração inválida: {exc}", file=sys.stderr)
        return 2

    try:
        info = sd.query_devices(args.device, "input")
        sd.check_input_settings(
            device=args.device, samplerate=16_000, channels=1, dtype="float32",
        )
        print(f"Microfone: {info['name']}")
        print(f"Whisper local: {args.model} (CPU/int8)")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")
    except (sd.PortAudioError, ValueError, OSError) as exc:
        print(f"Erro de microfone/Whisper: {exc}", file=sys.stderr)
        return 2

    client = BridgeClient(config)
    capture = AudioCapture(client, model, args.device)
    signal.signal(signal.SIGTERM, shutdown_signal)
    try:
        AgentRuntime(client, capture).run()
    except BridgeError as exc:
        print(f"Alexios não pôde se conectar: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Um sinal durante a inicialização também deve liberar os recursos.
        capture.shutdown()
        print("\nAlexios encerrado por Ctrl+C.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

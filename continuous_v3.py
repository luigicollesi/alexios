"""Alexios V3: Silero + Smart Turn + CAM++ e contexto compartilhado na web.

V3 é opt-in. O continuous.py V2 continua preservado e funcional.
Os endpoints /v3 só funcionarão após a implementação correspondente na web.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

from dotenv import load_dotenv

from agent.bridge import BridgeError
from agent.config import AgentConfig
from agent.runtime import AgentRuntime
from agent.v3.bridge import EventBridge
from agent.v3.capture import TurnAwareCapture
from agent.v3.dispatch import EventDispatcher
from agent.v3.models import CamPlusRegistry, SileroVad, SmartTurnDetector
from agent.v3.transcription import WhisperTurnTranscriber


def stop_signal(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Alexios V3: turnos e múltiplos locutores")
    parser.add_argument("--device", type=int, default=None, help="ID do microfone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", default="tiny", help="Whisper local: tiny/base (Pi 4: tiny)")
    parser.add_argument("--model-dir", default="models/v3", help="Diretório dos pesos ONNX")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--turn-threshold", type=float, default=0.5)
    parser.add_argument("--speaker-threshold", type=float, default=0.6)
    parser.add_argument("--model-threads", type=int, default=1)
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        config = AgentConfig.from_env()
        if args.model_threads < 1 or args.model_threads > 4:
            raise ValueError("--model-threads deve estar entre 1 e 4.")
        sd.check_input_settings(
            device=args.device, samplerate=16_000, channels=1, dtype="float32"
        )
        base = Path(args.model_dir)
        from faster_whisper import WhisperModel

        print("Carregando Silero VAD, Smart Turn, CAM++ e Whisper (CPU)...")
        vad = SileroVad(str(base / "silero_vad.onnx"), threshold=args.vad_threshold)
        classifier = SmartTurnDetector(
            str(base / "smart-turn-v3.2-cpu.onnx"),
            threshold=args.turn_threshold, threads=args.model_threads,
        )
        speakers = CamPlusRegistry(
            str(base / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
            threshold=args.speaker_threshold, threads=args.model_threads,
        )
        # Prioridade para menor carga no Pi 4. O modelo STT pode ser trocado
        # sem alterar os contratos da rede.
        stt_model = WhisperModel(args.model, device="cpu", compute_type="int8",
                                 cpu_threads=args.model_threads, num_workers=1)
        transcriber = WhisperTurnTranscriber(stt_model)
    except (ValueError, FileNotFoundError, OSError, RuntimeError, ImportError) as exc:
        print(f"Configuração V3 inválida: {exc}", file=sys.stderr)
        return 2

    client = EventBridge(config)
    try:
        # /v2/agent/poll registra a lease de presença; sync inicializa
        # a sequência V3 a partir da posição persistida no Neon.
        client.poll(-1)
        next_sequence = client.sync()
    except BridgeError as exc:
        print(f"Servidor não oferece o protocolo V3: {exc}. "
              "Use continuous.py para a V2 até o deploy do backend V3.",
              file=sys.stderr)
        return 1

    capture: TurnAwareCapture | None = None
    dispatcher = EventDispatcher(
        client, transcriber, speakers,
        next_sequence=next_sequence,
        enabled=lambda: capture is not None and capture.running,
    )
    capture = TurnAwareCapture(dispatcher, vad, classifier, args.device)
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        AgentRuntime(
            client, capture,
            mode_label="Alexios V3",
            capture_label="Silero + Smart Turn + CAM++ por turno",
        ).run()
    except BridgeError as exc:
        print(f"Falha de conexão do Agent V3: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        capture.shutdown()
        print("\nEncerrado por sinal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

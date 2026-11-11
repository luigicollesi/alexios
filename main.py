"""Cliente oficial Alexios: captura local, turnos e comunicação com a Web."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys

from dotenv import load_dotenv

from agent.bridge import BridgeError
from agent.config import AgentConfig
from agent.runtime import AgentRuntime
from agent.event_bridge import EventBridge
from agent.capture import TurnAwareCapture
from agent.dispatch import EventDispatcher
from agent.models import CamPlusRegistry, SileroVad, SmartTurnDetector
from agent.transcription import WhisperTurnTranscriber, load_vocabulary


def stop_signal(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(description="Alexios: cliente local de voz por turnos")
    parser.add_argument("--device", type=int, default=None, help="ID do microfone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--model", default="tiny", help="Modelo Whisper local (tiny, base, small, medium etc.)")
    parser.add_argument("--asr-context-mode", choices=("none", "hotwords", "prompt"),
                        default="hotwords",
                        help="Pistas de nomes/títulos; nunca substituição textual automática")
    parser.add_argument("--asr-vocabulary", type=Path,
                        default=Path(__file__).resolve().parent / "config" / "asr-vocabulary.json",
                        help="JSON de grafias relevantes (apenas carregamento local)")
    parser.add_argument("--asr-beam-size", type=int, default=5,
                        help="Busca do Whisper: 1 a 8 (padrão 5)")
    parser.add_argument("--model-dir", default="models/v3", help="Diretório dos pesos ONNX")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--turn-threshold", type=float, default=0.5)
    parser.add_argument("--speaker-threshold", type=float, default=0.6)
    parser.add_argument("--speaker-min-voice-seconds", type=float, default=0.8)
    parser.add_argument("--speaker-min-enroll-seconds", type=float, default=1.8)
    parser.add_argument("--speaker-near-margin", type=float, default=0.07)
    parser.add_argument("--speaker-distinct-max-similarity", type=float, default=0.20)
    parser.add_argument("--model-threads", type=int, default=1)
    parser.add_argument("--debug-audio", action="store_true",
                        help="Mostra volume, VAD, decisões do Smart Turn e etapas HTTP/ASR")
    parser.add_argument("--mic-test", type=int, metavar="SEGUNDOS",
                        help="Testa nível do microfone sem carregar modelos ou conectar à web")
    args = parser.parse_args()

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    if args.mic_test is not None:
        import numpy as np
        if not 1 <= args.mic_test <= 30:
            parser.error("--mic-test aceita de 1 a 30 segundos.")
        try:
            info = sd.query_devices(args.device, "input")
            print(f"[MIC] dispositivo: {info['name']}", flush=True)
            sd.check_input_settings(
                device=args.device, samplerate=16_000, channels=1,
                dtype="float32",
            )
            print(f"[MIC] gravando {args.mic_test}s sem enviar áudio à web...", flush=True)
            recorded = sd.rec(
                args.mic_test * 16_000, samplerate=16_000, channels=1,
                dtype="float32", device=args.device, blocking=True,
            ).reshape(-1)
            for second in range(args.mic_test):
                block = recorded[second * 16_000:(second + 1) * 16_000]
                rms = float(np.sqrt(np.mean(block * block)))
                peak = float(np.max(np.abs(block)))
                print(f"[MIC] segundo={second+1} rms={rms:.5f} pico={peak:.5f}",
                      flush=True)
            if float(np.max(np.abs(recorded))) < 0.003:
                print("[MIC] sinal muito baixo! Use --list-devices e informe --device.",
                      flush=True)
            return 0
        except Exception as exc:
            print(f"[MIC] erro de captura: {exc}", file=sys.stderr)
            return 2

    load_dotenv(Path(__file__).resolve().parent / ".env")
    try:
        config = AgentConfig.from_env()
        if args.model_threads < 1 or args.model_threads > 4:
            raise ValueError("--model-threads deve estar entre 1 e 4.")
        sd.check_input_settings(
            device=args.device, samplerate=16_000, channels=1, dtype="float32"
        )
        mic = sd.query_devices(args.device, "input")
        print(f"[MIC] entrada selecionada: {mic['name']} "
              f"({'padrão do sistema' if args.device is None else args.device})",
              flush=True)
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
            min_voice_seconds=args.speaker_min_voice_seconds,
            min_enroll_seconds=args.speaker_min_enroll_seconds,
            near_match_margin=args.speaker_near_margin,
            distinct_voice_max_similarity=args.speaker_distinct_max_similarity,
        )
        # Prioridade para menor carga no Pi 4. O modelo STT pode ser trocado
        # sem alterar os contratos da rede.
        stt_model = WhisperModel(args.model, device="cpu", compute_type="int8",
                                 cpu_threads=args.model_threads, num_workers=1)
        vocabulary = (load_vocabulary(args.asr_vocabulary)
                      if args.asr_context_mode != "none" else ())
        transcriber = WhisperTurnTranscriber(
            stt_model,
            vocabulary=vocabulary,
            context_mode=args.asr_context_mode,
            beam_size=args.asr_beam_size,
        )
        print(
            f"[ASR] modelo={args.model} idioma=pt beam={args.asr_beam_size} "
            f"contexto={args.asr_context_mode} termos={len(vocabulary)}",
            flush=True,
        )
    except (ValueError, FileNotFoundError, OSError, RuntimeError,
            ImportError, json.JSONDecodeError) as exc:
        print(f"Configuração do Alexios inválida: {exc}", file=sys.stderr)
        return 2

    client = EventBridge(config)
    try:
        # /v2/agent/poll registra a lease de presença; sync inicializa
        # a sequência V3 a partir da posição persistida no Neon.
        client.poll(-1)
        next_sequence = client.sync()
    except BridgeError as exc:
        print(f"Servidor não oferece o protocolo V3: {exc}. "
              "Verifique a configuração e os endpoints do servidor.",
              file=sys.stderr)
        return 1

    capture: TurnAwareCapture | None = None
    dispatcher = EventDispatcher(
        client, transcriber, speakers,
        next_sequence=next_sequence,
        enabled=lambda: capture is not None and capture.running,
        debug_audio=args.debug_audio,
    )
    capture = TurnAwareCapture(
        dispatcher, vad, classifier, args.device, debug_audio=args.debug_audio,
    )
    print(f"[ALEXIOS] conexão aceita; próximo evento={next_sequence}", flush=True)
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        AgentRuntime(
            client, capture,
            mode_label="Alexios",
            capture_label="Silero + Smart Turn + CAM++ por turno",
        ).run()
    except BridgeError as exc:
        print(f"Falha de conexão do Alexios: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        capture.shutdown()
        print("\nEncerrado por sinal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

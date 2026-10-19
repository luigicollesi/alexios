"""Alexios: primeira etapa de reconhecimento de voz local em português."""

import argparse
import queue
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000


def record_audio(device: int | None, max_seconds: int = 60) -> np.ndarray:
    """Grava áudio mono float32 enquanto o usuário não pressiona Enter."""
    chunks: queue.Queue[np.ndarray] = queue.Queue()
    errors = []

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcrição local por microfone com Faster-Whisper")
    parser.add_argument("--list-devices", action="store_true", help="Mostra os dispositivos disponíveis")
    parser.add_argument("--device", type=int, default=None, help="ID do dispositivo de entrada")
    parser.add_argument("--model", default="base", help="Modelo Whisper, por exemplo tiny, base ou small")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    try:
        info = sd.query_devices(args.device, "input")
        sd.check_input_settings(device=args.device, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        print(f"Microfone: {info['name']}")
        print(f"Carregando modelo '{args.model}' na CPU...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

        while True:
            audio = record_audio(args.device)
            if audio.size == 0 or np.max(np.abs(audio)) < 0.001:
                print("Nenhuma fala detectada. Tente novamente.")
            else:
                print("Transcrevendo...")
                segments, _ = model.transcribe(
                    audio, language="pt", beam_size=5, vad_filter=True
                )
                transcription = " ".join(
                    segment.text.strip() for segment in segments if segment.text.strip()
                )
                print(f"Você disse: {transcription or '[sem fala reconhecida]'}")

            if input("Nova gravação? [Enter = sim / n = sair]: ").strip().lower() in {"n", "não", "nao"}:
                break
    except (sd.PortAudioError, ValueError) as exc:
        print(f"Erro ao acessar o microfone: {exc}", file=sys.stderr)
        print("Execute 'python main.py --list-devices' para selecionar outro dispositivo.")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\nEncerrando.")


if __name__ == "__main__":
    main()

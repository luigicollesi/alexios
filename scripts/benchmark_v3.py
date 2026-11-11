"""Benchmark local da V3, sem conexão com Alexios Web.

Usa WAV PCM16 mono de 16 kHz, para avaliação em Raspberry Pi 4:
python scripts/benchmark_v3.py --wav ./sample.wav --whisper tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agent.models import CamPlusRegistry, SileroVad, SmartTurnDetector
from agent.transcription import WhisperTurnTranscriber


def wav16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate()) != (1, 2, 16000):
            raise ValueError("Requer WAV PCM16 mono a 16000 Hz.")
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768


def seconds(fn):
    before = time.perf_counter()
    result = fn()
    return round(1000 * (time.perf_counter() - before), 2), result


def main():
    parser = argparse.ArgumentParser(description="Benchmark do Alexios V3 (CPU)")
    parser.add_argument("--wav", type=Path, required=True, help="Exemplo de voz WAV PCM16/16k")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "v3")
    parser.add_argument("--whisper", default="", help="tiny/base; opcional")
    args = parser.parse_args()

    sample = wav16(args.wav)
    model_dir = args.model_dir
    result = {"duration_s": round(len(sample) / 16000, 2)}
    vad = SileroVad(str(model_dir / "silero_vad.onnx"))
    def feed_vad():
        n = 0
        for pos in range(0, len(sample) - vad.window_size + 1, vad.window_size):
            n += bool(vad.is_speech(sample[pos:pos + vad.window_size]))
        return n
    result["vad_total_ms"], result["vad_active_frames"] = seconds(feed_vad)

    smart = SmartTurnDetector(str(model_dir / "smart-turn-v3.2-cpu.onnx"))
    result["smart_turn_ms"], result["smart_turn_complete"] = seconds(lambda: smart.complete(sample))
    speaker = CamPlusRegistry(str(model_dir / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"))
    result["cam_plus_ms"], result["speaker"] = seconds(lambda: speaker.identify(sample))

    if args.whisper:
        from faster_whisper import WhisperModel
        model = WhisperModel(args.whisper, device="cpu", compute_type="int8", cpu_threads=1)
        stt = WhisperTurnTranscriber(model)
        result["whisper_ms"], result["transcript"] = seconds(lambda: stt.transcribe(sample))

    result["peak_rss_mb_linux"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

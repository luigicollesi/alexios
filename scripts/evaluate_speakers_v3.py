"""Avaliação local da identificação V3 sem rede nem persistência de biometria.

Manifest JSON (na ordem temporal, >=2 gravações por pessoa):
[
  {"wav": "clips/luigi-01.wav", "speaker": "pessoa_a"},
  {"wav": "clips/ana-01.wav", "speaker": "pessoa_b"}
]
Todas as WAVs devem ser PCM16 mono/16 kHz, gravadas com consentimento.
As métricas são agregadas; não são enviados áudio nem embeddings.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.models import CamPlusRegistry, SileroVad


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(),
                stream.getframerate()) != (1, 2, 16_000):
            raise ValueError(f"{path}: requer WAV PCM16 mono/16k")
        return np.frombuffer(stream.readframes(stream.getnframes()),
                             dtype="<i2").astype(np.float32) / 32768


def vad_voice_only(vad: SileroVad, samples: np.ndarray) -> np.ndarray:
    vad.reset()
    chunks: list[np.ndarray] = []
    size = vad.window_size
    for offset in range(0, len(samples) - size + 1, size):
        frame = samples[offset:offset + size]
        if vad.is_speech(frame):
            chunks.append(frame)
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(chunks)[-8 * 16_000:])


def summarize(rows: list[dict]) -> dict:
    by_truth: dict[str, set[str]] = defaultdict(set)
    by_pred: dict[str, set[str]] = defaultdict(set)
    unknown = 0
    duration_ms = []
    for row in rows:
        if row["predicted"] == "speaker_unknown":
            unknown += 1
        else:
            by_truth[row["speaker"]].add(row["predicted"])
            by_pred[row["predicted"]].add(row["speaker"])
        duration_ms.append(float(row["latency_ms"]))
    latency = np.asarray(duration_ms, dtype=np.float64)
    return {
        "turns": len(rows),
        "unknown_rate": round(unknown / len(rows), 4) if rows else 0,
        "fragmentation_extra_ids": sum(max(0, len(v) - 1) for v in by_truth.values()),
        "false_merge_extra_people": sum(max(0, len(v) - 1) for v in by_pred.values()),
        "identified_speakers": len(by_pred),
        "latency_p50_ms": round(float(np.percentile(latency, 50)), 2) if len(rows) else 0,
        "latency_p95_ms": round(float(np.percentile(latency, 95)), 2) if len(rows) else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliar o reconhecimento V3 sem acessar a web")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "v3")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-voice-seconds", type=float, default=0.8)
    parser.add_argument("--min-enroll-seconds", type=float, default=1.8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        parser.error("manifest deve ser lista JSON não vazia")
    for item in manifest:
        if not isinstance(item, dict) or not isinstance(item.get("wav"), str) or not isinstance(item.get("speaker"), str):
            parser.error("cada item precisa de 'wav' e 'speaker' strings")

    vad = SileroVad(str(args.model_dir / "silero_vad.onnx"))
    registry = CamPlusRegistry(
        str(args.model_dir / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
        threshold=args.threshold, min_voice_seconds=args.min_voice_seconds,
        min_enroll_seconds=args.min_enroll_seconds,
    )
    rows: list[dict] = []
    for item in manifest:
        wav_path = (args.manifest.parent / item["wav"]).resolve()
        original = read_wav(wav_path)
        start = time.perf_counter()
        voiced = vad_voice_only(vad, original)
        predicted = registry.identify(voiced)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        rows.append({
            "speaker": item["speaker"], "predicted": predicted,
            "latency_ms": elapsed, "voice_seconds": len(voiced) / 16_000,
        })
    print(json.dumps({
        "threshold": args.threshold,
        "min_voice_seconds": args.min_voice_seconds,
        "min_enroll_seconds": args.min_enroll_seconds,
        **summarize(rows),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark ASR local: WAV PCM16 16 kHz + referência humana.

Example:
 python scripts/benchmark_asr.py --manifest samples/asr-manifest.json --model base

Cada item do manifesto contém "wav" e "reference". O script não transmite
áudio/transcrição; imprime apenas métricas agregadas. A primeira execução de
um nome de modelo pode baixar os pesos via faster-whisper: execute antes do
teste, com a conexão e a licença apropriadas, não no loop de microfone.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time
import unicodedata
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.transcription import WhisperTurnTranscriber, load_vocabulary


def words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"\w+", normalized, re.UNICODE)


def edits(reference: list[str] | str, predicted: list[str] | str) -> int:
    """Levenshtein em espaço O(min(n, m))."""
    if len(predicted) > len(reference):
        reference, predicted = predicted, reference
    previous = list(range(len(predicted) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(predicted, 1):
            current.append(min(
                previous[j] + 1,
                current[-1] + 1,
                previous[j - 1] + (token != candidate),
            ))
        previous = current
    return previous[-1]


def phrase_count(tokens: list[str], phrase: list[str]) -> int:
    if not phrase:
        return 0
    return sum(tokens[pos:pos + len(phrase)] == phrase
               for pos in range(len(tokens) - len(phrase) + 1))


def summarize(records: list[dict], vocabulary: tuple[str, ...]) -> dict:
    total_words = total_chars = word_edits = char_edits = 0
    correct_terms = mentioned_terms = inserted_terms = 0
    times: list[float] = []
    for record in records:
        reference = words(record["reference"])
        predicted = words(record["predicted"])
        total_words += len(reference)
        word_edits += edits(reference, predicted)
        reference_chars = " ".join(reference)
        predicted_chars = " ".join(predicted)
        total_chars += len(reference_chars)
        char_edits += edits(reference_chars, predicted_chars)
        for term in vocabulary:
            lexical = words(term)
            actual = phrase_count(reference, lexical)
            found = phrase_count(predicted, lexical)
            mentioned_terms += actual
            correct_terms += min(actual, found)
            inserted_terms += max(0, found - actual)
        times.append(record["elapsed_ms"])
    latencies = np.asarray(times, dtype=np.float64)
    def rate(error_count: int, total: int) -> float | None:
        return round(error_count / total, 4) if total else None
    return {
        "utterances": len(records),
        "wer": rate(word_edits, total_words),
        "cer": rate(char_edits, total_chars),
        "name_title_recall": rate(correct_terms, mentioned_terms),
        "name_title_reference_mentions": mentioned_terms,
        "name_title_false_insertions": inserted_terms,
        "latency_p50_ms": (round(float(np.percentile(latencies, 50)), 2)
                           if len(latencies) else None),
        "latency_p95_ms": (round(float(np.percentile(latencies, 95)), 2)
                           if len(latencies) else None),
    }


def wav16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as stream:
        if (stream.getnchannels(), stream.getsampwidth(),
                stream.getframerate()) != (1, 2, 16_000):
            raise ValueError(f"{path}: esperado WAV PCM16 mono 16kHz")
        return np.frombuffer(stream.readframes(stream.getnframes()),
                             dtype="<i2").astype(np.float32) / 32768


def main() -> int:
    parser = argparse.ArgumentParser(description="Medir ASR local (sem upload)")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default="tiny", help="tiny, base ou small")
    parser.add_argument("--context-mode", choices=("none", "hotwords", "prompt"),
                        default="hotwords")
    parser.add_argument("--vocabulary", type=Path,
                        default=ROOT / "config" / "asr-vocabulary.json")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--model-threads", type=int, default=1)
    args = parser.parse_args()
    if args.model_threads not in range(1, 5):
        parser.error("--model-threads deve estar entre 1 e 4")
    if not 1 <= args.beam_size <= 8:
        parser.error("--beam-size deve estar entre 1 e 8")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (not isinstance(manifest, list) or not manifest
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("wav"), str)
                   or not isinstance(item.get("reference"), str)
                   or not item["reference"].strip() for item in manifest)):
        parser.error("manifest deve conter lista não vazia de WAVs e referências")
    vocabulary = load_vocabulary(args.vocabulary)
    # Term metrics remain comparable even with contextual bias disabled.
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=args.model_threads, num_workers=1)
    transcriber = WhisperTurnTranscriber(
        model, vocabulary=vocabulary, context_mode=args.context_mode,
        beam_size=args.beam_size, vad_filter=args.vad_filter,
    )
    results: list[dict] = []
    for item in manifest:
        wav_path = args.manifest.resolve().parent / item["wav"]
        audio = wav16(wav_path)
        started = time.perf_counter()
        output = transcriber.transcribe(audio)
        results.append({
            "reference": item["reference"],
            "predicted": output,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        })
    print(json.dumps({
        "model": args.model,
        "context_mode": args.context_mode,
        "beam_size": args.beam_size,
        "vad_filter": args.vad_filter,
        "terms": len(vocabulary),
        **summarize(results, vocabulary),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

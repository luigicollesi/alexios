"""Baixa pesos ONNX somente sob comando explícito, de fontes oficiais.

python scripts/download_models.py
Arquivos grandes não são incluídos no repositório Git.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import urllib.request

TARGET = Path(__file__).resolve().parent.parent / "models" / "v3"

MODELS = {
    "silero_vad.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        None,
    ),
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx": (
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/"
        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        None,
    ),
    "smart-turn-v3.2-cpu.onnx": (
        "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/"
        "smart-turn-v3.2-cpu.onnx",
        "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f",
    ),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in MODELS.items():
        target = TARGET / name
        if target.exists() and target.stat().st_size > 1024:
            if expected is None or sha256(target) == expected:
                print(f"Já existe: {target}")
                continue
        staging = target.with_name(target.name + ".part")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Alexios-V3/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response, staging.open("wb") as out:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(chunk)
            if staging.stat().st_size < 1024:
                raise RuntimeError(f"Download insuficiente: {name}")
            if expected and sha256(staging) != expected:
                raise RuntimeError(f"Checksum inválido de {name}; arquivo recusado.")
            staging.replace(target)
            print(f"Modelo verificado: {target}")
        finally:
            staging.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

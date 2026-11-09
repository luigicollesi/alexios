"""Sample-accurate 4 s windows, emitted every 3 s (1 s overlap)."""

from dataclasses import dataclass
import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 4 * SAMPLE_RATE
HOP_SAMPLES = 3 * SAMPLE_RATE


@dataclass(frozen=True)
class AudioWindow:
    index: int
    start_ms: int
    samples: np.ndarray


class WindowCutter:
    def __init__(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.offset_samples = 0
        self.index = 0

    def push(self, samples: np.ndarray) -> list[AudioWindow]:
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.buffer = np.concatenate((self.buffer, audio))
        ready: list[AudioWindow] = []
        while self.buffer.size >= WINDOW_SAMPLES:
            ready.append(AudioWindow(
                index=self.index,
                start_ms=self.offset_samples * 1000 // SAMPLE_RATE,
                samples=self.buffer[:WINDOW_SAMPLES].copy(),
            ))
            self.buffer = self.buffer[HOP_SAMPLES:]
            self.offset_samples += HOP_SAMPLES
            self.index += 1
        return ready

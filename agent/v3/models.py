"""Adaptadores reais para modelos open-source executados em CPU.

Importações opcionais ocorrem apenas durante inicialização da V3. Os pesos
nunca são obtidos silenciosamente da rede durante captura.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

SAMPLE_RATE = 16000


def require_model(path: str | Path) -> str:
    target = Path(path).expanduser().resolve()
    if not target.is_file() or target.stat().st_size < 1024:
        raise FileNotFoundError(f"Modelo ONNX ausente/inválido: {target}. Consulte V3.md.")
    return str(target)


class SileroVad:
    """VAD streaming via sherpa-onnx, usando o Silero oficial."""

    def __init__(self, model_path: str, *, threshold: float = 0.5) -> None:
        import sherpa_onnx
        if not 0 < threshold < 1:
            raise ValueError("threshold do VAD fora do intervalo.")
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = require_model(model_path)
        config.silero_vad.threshold = threshold
        config.silero_vad.min_silence_duration = 0.15
        config.silero_vad.min_speech_duration = 0.1
        config.silero_vad.max_speech_duration = 45
        config.sample_rate = SAMPLE_RATE
        self.window_size: int = config.silero_vad.window_size
        self._config = config
        self._module = sherpa_onnx
        self.reset()

    def reset(self) -> None:
        # Um detector novo não herda estado recorrente de outro ciclo de captura.
        # sherpa-onnx mantém internamente a fala até surgir silêncio.
        # O limite anterior (3s) era incompatível com 100s de fala contínua
        # e provocava "circular-buffer.cc:Push Overflow!" após ~3 segundos.
        # 120s usa até ~7,7 MB de amostras float32 e cobre turnos longos.
        self.detector = self._module.VoiceActivityDetector(
            self._config, buffer_size_in_seconds=120
        )

    def is_speech(self, samples: np.ndarray) -> bool:
        if len(samples) != self.window_size:
            raise ValueError(f"Silero requer {self.window_size} amostras por bloco.")
        self.detector.accept_waveform(
            np.ascontiguousarray(samples, dtype=np.float32)
        )
        # O engine mantém seu próprio buffer; descartar frames de segmento do VAD.
        while not self.detector.empty():
            self.detector.pop()
        return bool(self.detector.is_speech_detected())


class SmartTurnDetector:
    """Smart Turn v3.2 ONNX com front-end Whisper log-mel (sem PyTorch)."""

    def __init__(self, model_path: str, *, threshold: float = 0.5, threads: int = 1) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor
        if not 0 < threshold < 1:
            raise ValueError("threshold Smart Turn fora do intervalo.")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = max(1, threads)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            require_model(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.features = WhisperFeatureExtractor(chunk_length=8)
        self.threshold = threshold

    def complete(self, recent_audio: np.ndarray) -> bool:
        audio = np.asarray(recent_audio[-8 * SAMPLE_RATE:], dtype=np.float32)
        inputs = self.features(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=8 * SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        features = np.asarray(inputs.input_features, dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: features})
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        if not np.isfinite(probability) or probability < 0 or probability > 1:
            raise ValueError("Saída inválida do Smart Turn; probabilidades esperadas.")
        return probability >= self.threshold


class CamPlusRegistry:
    """Identificação conservadora e efêmera de locutores com CAM++.

    O áudio recebido já contém apenas frames marcados como fala pelo VAD.
    Nunca transforma uma gravação curta/ruidosa em novo speaker. Um candidato
    adicional precisa de duas amostras independentes de boa qualidade.
    Thresholds são hipóteses de engenharia a calibrar com áudio real.
    """

    def __init__(
        self, model_path: str, *, threshold: float = 0.6,
        min_voice_seconds: float = 0.8, min_enroll_seconds: float = 1.8,
        threads: int = 1, near_match_margin: float = 0.07,
        candidate_ttl_s: float = 180.0,
    ) -> None:
        import sherpa_onnx
        if not 0 < threshold < 1:
            raise ValueError("threshold de locutor fora do intervalo.")
        if min_voice_seconds < 0.4 or min_enroll_seconds < min_voice_seconds:
            raise ValueError("Durações de verificação/cadastro inválidas.")
        if not 0 < near_match_margin < threshold:
            raise ValueError("Margem de identificação inválida.")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=require_model(model_path),
            num_threads=max(1, threads), provider="cpu", debug=False,
        )
        if not config.validate():
            raise ValueError("Configuração do CAM++ inválida.")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.manager = sherpa_onnx.SpeakerEmbeddingManager(self.extractor.dim)
        self.threshold = threshold
        self.min_samples = round(min_voice_seconds * SAMPLE_RATE)
        self.enroll_samples = round(min_enroll_seconds * SAMPLE_RATE)
        self.near_match_margin = near_match_margin
        self.candidate_ttl_s = candidate_ttl_s
        self.counter = 0
        # Embeddings somente em memória da sessão, nunca em logs/arquivos.
        self.references: dict[str, np.ndarray] = {}
        self.candidate: tuple[np.ndarray, float] | None = None
        self.last_decision: dict[str, str | float] = {"reason": "not_started"}

    @staticmethod
    def _similarity(left: np.ndarray, right: np.ndarray) -> float:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-10:
            return 0.0
        return float(np.dot(left, right) / denominator)

    def _decision(self, reason: str, voice_seconds: float,
                  best_similarity: float | None = None) -> str:
        # Only structural metadata: never raw audio or biometric vectors.
        self.last_decision = {
            "reason": reason,
            "voice_seconds": round(voice_seconds, 2),
            **({"best_similarity": round(best_similarity, 3)}
               if best_similarity is not None else {}),
        }
        return "speaker_unknown"

    def identify(self, samples: np.ndarray) -> str:
        """Find a speaker or defer enrollment until adequate voice evidence."""
        import time

        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        duration = len(samples) / SAMPLE_RATE
        if len(samples) < self.min_samples or not np.isfinite(samples).all():
            return self._decision("insufficient_voice", duration)
        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        peak = float(np.max(np.abs(samples)))
        clipping = float(np.mean(np.abs(samples) >= 0.99))
        if rms < 0.006 or peak > 1.1 or clipping > 0.03:
            return self._decision("low_quality_voice", duration)

        voice = np.ascontiguousarray(samples[-8 * SAMPLE_RATE:])
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=voice)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return self._decision("extractor_not_ready", duration)
        embedding = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        if embedding.shape != (self.extractor.dim,) or not np.isfinite(embedding).all():
            return self._decision("invalid_embedding", duration)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-10:
            return self._decision("zero_embedding", duration)
        normalized = embedding / norm

        # Preserve the known sherpa-onnx search behavior and its threshold.
        matched = self.manager.search(embedding, threshold=self.threshold)
        if matched:
            self.candidate = None
            self.last_decision = {
                "reason": "recognized", "voice_seconds": round(duration, 2),
            }
            return str(matched)

        similarity = max((
            self._similarity(normalized, prior)
            for prior in self.references.values()
        ), default=0.0)
        if len(samples) < self.enroll_samples:
            return self._decision("short_unmatched", duration, similarity)

        # A borderline match is ambiguous; do not create a duplicate speaker.
        if self.references and similarity >= self.threshold - self.near_match_margin:
            return self._decision("near_existing_unknown", duration, similarity)

        now = time.monotonic()
        if self.references:
            if self.candidate is None or now - self.candidate[1] > self.candidate_ttl_s:
                self.candidate = (normalized.copy(), now)
                return self._decision("awaiting_second_sample", duration, similarity)
            candidate, _ = self.candidate
            if self._similarity(normalized, candidate) < self.threshold + 0.03:
                self.candidate = (normalized.copy(), now)
                return self._decision("candidate_changed", duration, similarity)

        speaker_id = f"speaker_{self.counter + 1:02d}"
        if not self.manager.add(speaker_id, embedding):
            raise RuntimeError("CAM++ não conseguiu cadastrar o embedding.")
        self.counter += 1
        self.references[speaker_id] = normalized.copy()
        self.candidate = None
        self.last_decision = {
            "reason": "enrolled", "voice_seconds": round(duration, 2),
        }
        return speaker_id

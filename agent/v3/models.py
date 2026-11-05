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
        config.silero_vad.max_speech_duration = 100
        config.sample_rate = SAMPLE_RATE
        self.window_size: int = config.silero_vad.window_size
        self._config = config
        self._module = sherpa_onnx
        self.reset()

    def reset(self) -> None:
        # Um detector novo não herda estado recorrente de outro ciclo de captura.
        self.detector = self._module.VoiceActivityDetector(
            self._config, buffer_size_in_seconds=3
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
    """Diferencia locutores da sessão; sem nome real nem persistência biométrica."""

    def __init__(
        self, model_path: str, *, threshold: float = 0.6,
        min_voice_seconds: float = 0.8, threads: int = 1,
    ) -> None:
        import sherpa_onnx
        if not 0 < threshold < 1:
            raise ValueError("threshold de locutor fora do intervalo.")
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
        self.counter = 0

    def identify(self, samples: np.ndarray) -> str:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) < self.min_samples:
            return "speaker_unknown"
        # Não é seguro identificar misturas de vozes. O modo inicial pressupõe
        # um locutor dominante por turno; sobreposição exigirá diarização extra.
        # Uma janela curta limita o custo do CAM++ em turnos de até 45s.
        voice = samples[-8 * SAMPLE_RATE:]
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE,
                               waveform=np.ascontiguousarray(voice))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return "speaker_unknown"
        embedding = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        speaker = self.manager.search(embedding, threshold=self.threshold)
        if speaker:
            return str(speaker)
        self.counter += 1
        speaker_id = f"speaker_{self.counter:02d}"
        if not self.manager.add(speaker_id, embedding):
            raise RuntimeError("CAM++ não conseguiu cadastrar o embedding.")
        return speaker_id

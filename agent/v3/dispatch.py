"""Despacho serial dos eventos de conversa por instância.

A fila de TURN_END e FLOOR_OPEN é FIFO: o segundo jamais ultrapassa um
turno que ainda espera Whisper. O canal ACTIVITY é prioritário e independente,
permitindo invalidar uma resposta quando alguém volta a falar.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import queue
import sys
import threading
import time
from uuid import uuid4

from agent.bridge import BridgeError
from .turns import CapturedTurn, FloorOpened


@dataclass(frozen=True)
class _Pending:
    kind: str
    sequence: int
    event_id: str
    payload: dict


class EventDispatcher:
    def __init__(
        self,
        client: object,
        transcriber: object,
        speakers: object,
        *,
        next_sequence: int,
        max_pending: int = 8,
        enabled: Callable[[], bool] | None = None,
        debug_audio: bool = False,
    ) -> None:
        if next_sequence < 1:
            raise ValueError("Sequência inicial inválida.")
        self.client = client
        self.transcriber = transcriber
        self.speakers = speakers
        self.sequence = next_sequence
        self.enabled = enabled or (lambda: True)
        self.debug_audio = debug_audio
        # Não abrir piso de fala após transcrição vazia/ruído.
        self.last_confirmed_turn_epoch = -1
        self.events: queue.Queue[CapturedTurn | FloorOpened] = queue.Queue(maxsize=max_pending)
        self.activity: queue.Queue[int] = queue.Queue(maxsize=16)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_epoch = 0
        self.acknowledged_activity = 0
        self.failure: str | None = None
        self.sender = threading.Thread(target=self._deliver_loop, name="alexios-v3-events", daemon=True)
        self.activity_sender = threading.Thread(
            target=self._activity_loop, name="alexios-v3-activity", daemon=True,
        )
        self.sender.start()
        self.activity_sender.start()

    @property
    def activity_epoch(self) -> int:
        with self.lock:
            return self.latest_epoch

    def speech_started(self, epoch: int) -> None:
        with self.lock:
            if epoch <= self.latest_epoch:
                return
            self.latest_epoch = epoch
        try:
            self.activity.put_nowait(epoch)
        except queue.Full as exc:
            self.failure = "Fila prioritária de atividade esgotada."
            raise RuntimeError(self.failure) from exc

    def cancel_floor(self) -> None:
        # Encerramento FIM/recaptura invalida todas as aberturas pendentes.
        with self.lock:
            self.latest_epoch += 1

    def enqueue(self, event: CapturedTurn | FloorOpened) -> None:
        if self.stop_event.is_set() or self.failure:
            raise RuntimeError(self.failure or "Despachante encerrado.")
        try:
            self.events.put_nowait(event)
        except queue.Full as exc:
            self.failure = "Fila de turnos lotada; captura deve parar para evitar perda silenciosa."
            raise RuntimeError(self.failure) from exc

    def _valid_floor(self, event: FloorOpened) -> bool:
        with self.lock:
            return (event.activity_epoch == self.latest_epoch
                    and self.acknowledged_activity >= event.activity_epoch
                    and self.enabled() and not self.stop_event.is_set())

    def _activity_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                epoch = self.activity.get(timeout=0.2)
            except queue.Empty:
                continue
            delay = 1
            try:
                while not self.stop_event.is_set():
                    try:
                        self.client.announce_activity(epoch)
                        with self.lock:
                            self.acknowledged_activity = max(self.acknowledged_activity, epoch)
                        if self.debug_audio:
                            print(f"[HTTP V3] activity confirmado epoch={epoch}", flush=True)
                        break
                    except BridgeError as exc:
                        if exc.fatal:
                            self.failure = str(exc)
                            break
                        time.sleep(delay)
                        delay = min(delay * 2, 10)
            finally:
                self.activity.task_done()

    def _prepare(self, event: CapturedTurn | FloorOpened) -> _Pending | None:
        sequence = self.sequence
        event_id = str(uuid4())
        if isinstance(event, FloorOpened):
            # A V3 só deve avaliar uma resposta após texto de turno concluído.
            # Isso evita a IA falar sobre ruído, mesmo quando o VAD detecta voz.
            if self.last_confirmed_turn_epoch != event.activity_epoch:
                print("[V3] floor_open ignorado: nenhum turn_end transcrito neste turno.",
                      flush=True)
                return None
            # O sinal urgente de início de fala deve estar confirmado antes de
            # uma abertura ser autorizada. Não descartar a abertura só porque
            # a confirmação ainda está viajando pela rede.
            limit = time.monotonic() + 12
            while time.monotonic() < limit and not self.stop_event.is_set():
                with self.lock:
                    obsolete = event.activity_epoch != self.latest_epoch
                    ack = self.acknowledged_activity >= event.activity_epoch
                if obsolete or not self.enabled():
                    return None
                if ack:
                    break
                time.sleep(0.05)
            if not self._valid_floor(event):
                # Não permite uma avaliação tardia sem confirmação de atividade.
                return None
            return _Pending("floor_open", sequence, event_id, {
                "eventId": event_id,
                "sequence": sequence,
                "activityEpoch": event.activity_epoch,
                "silenceMs": event.silence_ms,
                "afterSequence": sequence - 1,
            })
        # Não armazenamos contexto por locutor. O servidor mantém uma única
        # memória por instanceId; apenas a etiqueta speakerId é transmitida.
        started = time.monotonic()
        text = self.transcriber.transcribe(event.audio)
        elapsed = time.monotonic() - started
        if not text:
            print(f"[WHISPER] transcrição vazia ({elapsed:.1f}s); nada enviado.",
                  flush=True)
            return None
        print(f"[WHISPER] {len(text)} caracteres em {elapsed:.1f}s.", flush=True)
        if self.debug_audio:
            print(f"[WHISPER] texto: {text[:250]}", flush=True)
        voice = event.speaker_audio if event.speaker_audio is not None else event.audio
        speaker_id = self.speakers.identify(voice)
        if self.debug_audio:
            print(
                f"[CAM++] locutor={speaker_id} "
                f"voz_util={len(voice) / 16000:.2f}s "
                f"turno_total={len(event.audio) / 16000:.2f}s "
                f"qualidade={getattr(self.speakers, 'last_decision', {})}",
                flush=True,
            )
        kind = "turn_end" if event.complete else "turn_chunk"
        return _Pending(kind, sequence, event_id, {
            "eventId": event_id,
            "turnId": event.turn_id,
            "sequence": sequence,
            "speakerId": speaker_id,
            "transcript": text,
            "startedAtMs": event.started_ms,
            "endedAtMs": event.ended_ms,
            "activityEpoch": event.activity_epoch,
        })

    def _deliver_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                event = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                try:
                    prepared = self._prepare(event)
                except Exception as exc:
                    # Falhas de transcrição/locutor não podem descartar turnos
                    # silenciosamente. Paramos a ingestão e registramos o erro.
                    self.failure = f"Processamento de turno falhou: {exc}"
                    print(self.failure, file=sys.stderr)
                    break
                if prepared is None:
                    continue
                delay = 1
                while not self.stop_event.is_set():
                    if isinstance(event, FloorOpened) and not self._valid_floor(event):
                        if self.debug_audio:
                            print("[V3] floor_open invalidado por nova fala ou microfone inativo.",
                                  flush=True)
                        break
                    try:
                        self.client.send_event(prepared.kind, prepared.payload)
                        if isinstance(event, CapturedTurn) and event.complete:
                            self.last_confirmed_turn_epoch = event.activity_epoch
                        print(f"[HTTP V3] {prepared.kind} confirmado "
                              f"seq={self.sequence} epoch={event.activity_epoch}",
                              flush=True)
                        self.sequence += 1
                        break
                    except BridgeError as exc:
                        if exc.fatal or exc.conflict:
                            self.failure = f"Evento V3 rejeitado: {exc}"
                            break
                        print(f"V3: reenviando {prepared.kind}: {exc}", file=sys.stderr)
                        time.sleep(delay)
                        delay = min(delay * 2, 10)
                if self.failure:
                    break
            finally:
                self.events.task_done()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.sender.join(timeout=3)
        self.activity_sender.join(timeout=3)
        if not self.events.empty():
            print("Aviso: eventos V3 não confirmados antes do encerramento. "
                  "Fila V3 ainda não possui armazenamento durável.", file=sys.stderr)

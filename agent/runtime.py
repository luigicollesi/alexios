"""Controla presença, comandos remotos e ciclo de vida do microfone."""

from __future__ import annotations

import sys
import time
from typing import Protocol

from .bridge import BridgeClient, BridgeError, PollResult

OFFLINE_PAUSE_AFTER = 35
MAX_RETRY_SECONDS = 10


class CaptureDevice(Protocol):
    @property
    def running(self) -> bool: ...
    @property
    def last_error(self) -> str | None: ...
    def start(self) -> bool: ...
    def stop(self) -> None: ...
    def shutdown(self) -> None: ...


class AgentRuntime:
    def __init__(
        self, client: BridgeClient, capture: CaptureDevice, *,
        mode_label: str = "Alexios",
        capture_label: str = "Silero + Smart Turn + CAM++ por turno",
    ) -> None:
        self.mode_label = mode_label
        self.capture_label = capture_label
        self.client = client
        self.capture = capture
        self.command_seq = -1
        self.last_success = time.monotonic()
        self.retry_delay = 1
        self.last_reported: str | None = None

    def _report(self, state: str, *, timeout: int = 15) -> bool:
        try:
            self.client.report(state, timeout=timeout)
            self.last_reported = state
            return True
        except BridgeError as exc:
            print(f"Falha ao confirmar estado {state}: {exc}", file=sys.stderr)
            if exc.fatal:
                raise
            return False

    def _stop_capture(self) -> None:
        if self.capture.running:
            # A gravação é encerrada localmente mesmo se a rede falhar.
            self.capture.stop()
            print("Microfone desligado; long polling permanece conectado.")
        if self.last_reported != "stopped":
            self._report("stopped")

    def apply_command(self, command: PollResult) -> None:
        self.command_seq = command.command_seq
        if not command.capture_enabled:
            self._stop_capture()
            return
        if self.capture.running:
            return

        if not self._report("starting"):
            return
        if self.capture.start():
            try:
                if self._report("running"):
                    print(f"ESCUTA ATIVA: {self.capture_label}.")
                    return
            except BridgeError:
                self.capture.stop()
                raise
            self.capture.stop()
            self._report("stopped")
            return

        self._report("stopped")
        error = self.capture.last_error or "Dispositivo de entrada indisponível."
        print(f"Não foi possível iniciar o microfone: {error}", file=sys.stderr)

    def run(self) -> None:
        print(f"{self.mode_label}: presença e comandos via long polling; IA exclusiva na web.")
        try:
            while True:
                try:
                    command = self.client.poll(self.command_seq)
                except BridgeError as exc:
                    if exc.fatal:
                        raise
                    print(f"Long polling: {exc}", file=sys.stderr)
                    if self.capture.running and time.monotonic() - self.last_success > OFFLINE_PAUSE_AFTER:
                        self.capture.stop()
                        self.last_reported = None
                        print("Sem conexão: microfone interrompido preventivamente.", file=sys.stderr)
                    time.sleep(self.retry_delay)
                    self.retry_delay = min(self.retry_delay * 2, MAX_RETRY_SECONDS)
                    continue
                self.last_success = time.monotonic()
                self.retry_delay = 1
                self.apply_command(command)
        except KeyboardInterrupt:
            print("\nEncerrando Alexios e long polling.")
        finally:
            self.capture.shutdown()
            # Pode falhar após desconexão; o servidor também expira a lease em 45 s.
            try:
                self.client.report("stopped", timeout=5)
            except BridgeError:
                pass
            print("Microfone e long polling finalizados.")

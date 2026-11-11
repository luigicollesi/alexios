"""Comunicação HTTP compartilhada: presença e comandos de captura.

O servidor mantém os endpoints /v2/agent/poll e /v2/agent/state
por compatibilidade de protocolo. A conversação utiliza /v3/agent/*.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
from typing import Any
import urllib.error
import urllib.request
from uuid import uuid4

from .config import AgentConfig

POLL_TIMEOUT_SECONDS = 32
REQUEST_TIMEOUT_SECONDS = 15


class BridgeError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def fatal(self) -> bool:
        return self.status in {400, 401, 403, 404, 413, 422}

    @property
    def conflict(self) -> bool:
        return self.status == 409


@dataclass(frozen=True)
class PollResult:
    command_seq: int
    capture_enabled: bool


class BridgeClient:
    def __init__(self, config: AgentConfig, *, run_id: str | None = None) -> None:
        self.url = config.bridge_url
        self.token = config.agent_token
        self.instance_id = config.instance_id
        self.run_id = run_id or str(uuid4())

    def _request(self, path: str, body: dict[str, Any], *, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict:
        payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            self.url + path,
            data=payload,
            method="POST",
            headers={
                "Authorization": "Bearer " + self.token,
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise BridgeError(f"HTTP {exc.code} em {path}", status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise BridgeError(f"Falha de comunicação em {path}: {exc}") from exc
        except (ValueError, UnicodeDecodeError) as exc:
            raise BridgeError(f"Resposta JSON inválida de {path}") from exc
        if not isinstance(data, dict):
            raise BridgeError(f"Resposta inesperada de {path}")
        return data

    def _identity(self) -> dict[str, str]:
        return {"instanceId": self.instance_id, "runId": self.run_id}

    def poll(self, after: int) -> PollResult:
        if type(after) is not int or after < -1:
            raise ValueError("Cursor de comando inválido.")
        result = self._request(
            "/v2/agent/poll", {**self._identity(), "after": after},
            timeout=POLL_TIMEOUT_SECONDS,
        )
        command = result.get("commandSeq")
        enabled = result.get("captureEnabled")
        if type(command) is not int or command < 0 or type(enabled) is not bool:
            raise BridgeError("Resposta inválida do long polling V2.")
        return PollResult(command, enabled)

    def report(self, power: str, *, timeout: int = REQUEST_TIMEOUT_SECONDS) -> None:
        if power not in {"starting", "running", "stopping", "stopped"}:
            raise ValueError("Estado de microfone inválido.")
        self._request(
            "/v2/agent/state",
            {**self._identity(), "power": power},
            timeout=timeout,
        )


class EventBridge(BridgeClient):
    def sync(self) -> int:
        result = self._request("/v3/agent/sync", self._identity())
        sequence = result.get("nextSequence")
        if type(sequence) is not int or sequence < 1:
            raise BridgeError("Handshake V3 inválido: nextSequence esperado.")
        return sequence

    def send_event(self, kind: str, event: dict[str, Any]) -> None:
        if kind not in {"turn_end", "turn_chunk", "floor_open"}:
            raise ValueError("Tipo de evento V3 não autorizado.")
        endpoint = {
            "turn_end": "/v3/agent/turn",
            "turn_chunk": "/v3/agent/chunk",
            "floor_open": "/v3/agent/floor",
        }[kind]
        response = self._request(endpoint, {**self._identity(), **event})
        if response.get("accepted") is not True:
            raise BridgeError(f"Evento V3 não confirmado ({kind}).")

    def announce_activity(self, epoch: int) -> None:
        result = self._request("/v3/agent/activity", {
            **self._identity(),
            "state": "speech_started",
            "activityEpoch": epoch,
        }, timeout=6)
        if result.get("accepted") is not True:
            raise BridgeError("Atividade V3 não confirmada.")

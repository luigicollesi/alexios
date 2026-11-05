"""Transporte de eventos V3. Reutiliza o canal de presença e comandos V2.

O servidor V3 ainda deve implementar /v3/agent/sync|turn|chunk|floor|activity
antes da integração real. Nenhuma chamada à API de IA é feita localmente.
"""

from __future__ import annotations

from typing import Any

from agent.bridge import BridgeClient, BridgeError


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

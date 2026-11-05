"""Configuração do agente V2; nenhuma credencial é impressa em logs."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from urllib.parse import urlsplit

_INSTANCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{20,80}$")


@dataclass(frozen=True)
class AgentConfig:
    bridge_url: str
    agent_token: str
    instance_id: str

    @classmethod
    def from_env(cls) -> "AgentConfig":
        url = os.getenv("ALEXIOS_BRIDGE_URL", "https://alexios-web.vercel.app").strip().rstrip("/")
        token = os.getenv("ALEXIOS_AGENT_TOKEN", "").strip()
        instance = os.getenv("ALEXIOS_INSTANCE_ID", "").strip()

        parts = urlsplit(url)
        local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
        if not parts.netloc or (parts.scheme != "https" and not (local and parts.scheme == "http")):
            raise ValueError("ALEXIOS_BRIDGE_URL deve usar HTTPS (HTTP apenas localhost).")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("ALEXIOS_BRIDGE_URL não pode conter credenciais, query ou fragmento.")
        if len(token) < 24:
            raise ValueError("ALEXIOS_AGENT_TOKEN ausente ou com menos de 24 caracteres.")
        if not _INSTANCE_PATTERN.fullmatch(instance):
            raise ValueError(
                "Configure ALEXIOS_INSTANCE_ID fixo com 20 a 80 caracteres "
                "[A-Za-z0-9_-]. Gere: openssl rand -hex 24"
            )
        return cls(url, token, instance)

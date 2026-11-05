"""Contratos HTTP V3: instância compartilhada e idempotência por eventId."""

import io
import json
import unittest
from unittest.mock import patch

from agent.config import AgentConfig
from agent.v3.bridge import EventBridge
from agent.bridge import BridgeError


class V3BridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = EventBridge(
            AgentConfig("https://alexios-web.vercel.app", "a" * 32, "alx_" + "f" * 48),
            run_id="b30d0001-a6ab-4000-bbbb-112233445566",
        )
        self.calls = []

    def api(self, result):
        def respond(request, timeout):
            self.calls.append((request.full_url, json.loads(request.data), timeout))
            return io.BytesIO(json.dumps(result).encode())
        return respond

    def test_sync_uses_shared_instance(self):
        with patch("agent.bridge.urllib.request.urlopen", self.api({"nextSequence": 12})):
            seq = self.bridge.sync()
        self.assertEqual(seq, 12)
        url, body, _ = self.calls[-1]
        self.assertTrue(url.endswith("/v3/agent/sync"))
        self.assertEqual(body["instanceId"], self.bridge.instance_id)
        self.assertEqual(body["runId"], self.bridge.run_id)

    def test_turn_and_floor_routes_do_not_ask_ia_locally(self):
        with patch("agent.bridge.urllib.request.urlopen", self.api({"accepted": True})):
            self.bridge.send_event("turn_end", {
                "eventId": "a"*36, "sequence": 9,
                "speakerId": "speaker_01", "transcript": "Olá",
            })
            self.bridge.send_event("floor_open", {
                "eventId": "b"*36, "sequence": 10, "afterSequence": 9,
            })
            self.bridge.announce_activity(2)
        self.assertTrue(self.calls[0][0].endswith("/v3/agent/turn"))
        self.assertTrue(self.calls[1][0].endswith("/v3/agent/floor"))
        self.assertTrue(self.calls[2][0].endswith("/v3/agent/activity"))
        for _, payload, _ in self.calls:
            self.assertEqual(payload["instanceId"], self.bridge.instance_id)
            self.assertEqual(payload["runId"], self.bridge.run_id)

    def test_missing_ack_rejected(self):
        with patch("agent.bridge.urllib.request.urlopen", self.api({})):
            with self.assertRaises(BridgeError):
                self.bridge.send_event("turn_end", {"sequence": 1})


if __name__ == "__main__":
    unittest.main()

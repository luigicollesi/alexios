"""Contract tests with a simulated HTTPS server response; no external credentials."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from agent.bridge import BridgeClient, BridgeError, PollResult
from agent.config import AgentConfig


class BridgeV2Tests(unittest.TestCase):
    def setUp(self):
        self.client = BridgeClient(
            AgentConfig(
                "https://alexios-web.vercel.app",
                "a" * 32,
                "alx_" + "1" * 48,
            ),
            run_id="85b50a25-32da-4c2a-ae34-129bb2acb454",
        )
        self.calls = []

    def mock_api(self, data):
        def respond(request, timeout):
            self.calls.append((request, timeout))
            return io.BytesIO(json.dumps(data).encode("utf-8"))
        return respond

    def test_poll_contains_presence_and_commands_only(self):
        with patch("agent.bridge.urllib.request.urlopen", self.mock_api(
            {"commandSeq": 7, "captureEnabled": False}
        )):
            poll = self.client.poll(6)
        self.assertEqual(poll, PollResult(7, False))
        req, timeout = self.calls[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertTrue(req.full_url.endswith("/v2/agent/poll"))
        body = json.loads(req.data)
        self.assertEqual(body["after"], 6)
        self.assertEqual(body["runId"], self.client.run_id)
        self.assertEqual(body["instanceId"], self.client.instance_id)
        self.assertEqual(set(body), {"after", "instanceId", "runId"})
        self.assertEqual(timeout, 32)
        self.assertNotIn("speech", body)

    def test_report_microphone_power(self):
        with patch("agent.bridge.urllib.request.urlopen", self.mock_api({"ok": True})):
            self.client.report("stopped")
        req, _ = self.calls[0]
        self.assertTrue(req.full_url.endswith("/v2/agent/state"))
        self.assertEqual(json.loads(req.data)["power"], "stopped")

    def test_upload_has_same_identity_and_window(self):
        words = [{"text": "olá", "startMs": 5, "endMs": 300}]
        with patch("agent.bridge.urllib.request.urlopen", self.mock_api(
            {"acceptedWords": 1, "windowIndex": 4}
        )):
            accepted = self.client.upload("12345678-1234-4321-abcd-0123456789ab", 4, words)
        self.assertEqual(accepted, 1)
        req, _ = self.calls[0]
        data = json.loads(req.data)
        self.assertTrue(req.full_url.endswith("/v2/transcript"))
        self.assertEqual(data["words"], words)
        self.assertEqual(data["windowIndex"], 4)
        self.assertEqual(data["runId"], self.client.run_id)

    def test_unauthorized_is_fatal_but_lease_conflict_is_retryable(self):
        def error(status):
            raise HTTPError(
                self.client.url + "/v2/agent/poll", status, "error", {},
                io.BytesIO(b'{"error":"invalid"}'),
            )
        with patch("agent.bridge.urllib.request.urlopen", side_effect=lambda *_args, **_kwargs: error(401)):
            with self.assertRaises(BridgeError) as caught:
                self.client.poll(-1)
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(caught.exception.status, 401)
        with patch("agent.bridge.urllib.request.urlopen", side_effect=lambda *_args, **_kwargs: error(409)):
            with self.assertRaises(BridgeError) as caught:
                self.client.poll(-1)
        self.assertFalse(caught.exception.fatal)
        self.assertTrue(caught.exception.conflict)

    def test_rejects_incomplete_poll_payload(self):
        with patch("agent.bridge.urllib.request.urlopen", self.mock_api(
            {"commandSeq": 0, "speech": "não esperado"}
        )):
            with self.assertRaises(BridgeError):
                self.client.poll(-1)


if __name__ == "__main__":
    unittest.main()

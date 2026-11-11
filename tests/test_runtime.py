"""State-machine tests without microphone hardware or a live server."""
import unittest
from unittest.mock import patch

from agent.bridge import BridgeError, PollResult
from agent.runtime import AgentRuntime


class FakeCapture:
    def __init__(self):
        self.running = False
        self.last_error = None
        self.calls = []

    def start(self):
        self.calls.append("start")
        self.running = True
        return True

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def shutdown(self):
        self.calls.append("shutdown")
        self.running = False


class FakeClient:
    def __init__(self, polls=()):
        self.polls = iter(polls)
        self.reports = []
        self.cursor = []

    def report(self, power, *, timeout=15):
        self.reports.append((power, timeout))

    def poll(self, after):
        self.cursor.append(after)
        value = next(self.polls)
        if isinstance(value, Exception):
            raise value
        if value is KeyboardInterrupt:
            raise KeyboardInterrupt
        return value


class RuntimeTests(unittest.TestCase):
    def test_start_stop_and_restart_does_not_close_polling(self):
        capture = FakeCapture()
        client = FakeClient()
        runtime = AgentRuntime(client, capture)
        runtime.apply_command(PollResult(0, True))
        self.assertTrue(capture.running)
        self.assertEqual([x[0] for x in client.reports], ["starting", "running"])
        runtime.apply_command(PollResult(1, False))
        self.assertFalse(capture.running)
        self.assertEqual(client.reports[-1][0], "stopped")
        runtime.apply_command(PollResult(1, False))
        self.assertEqual([x[0] for x in client.reports].count("stopped"), 1)
        runtime.apply_command(PollResult(2, True))
        self.assertTrue(capture.running)
        self.assertEqual(capture.calls, ["start", "stop", "start"])

    def test_polling_keeps_running_while_microphone_is_off(self):
        client = FakeClient([
            PollResult(0, False),
            PollResult(0, False),
            KeyboardInterrupt,
        ])
        capture = FakeCapture()
        runtime = AgentRuntime(client, capture)
        runtime.run()
        self.assertEqual(client.cursor, [-1, 0, 0])
        self.assertEqual(capture.calls, ["shutdown"])
        self.assertEqual(client.reports[-1], ("stopped", 5))

    def test_lease_conflict_retries_without_capture(self):
        client = FakeClient([
            BridgeError("HTTP 409", status=409),
            PollResult(0, False),
            KeyboardInterrupt,
        ])
        capture = FakeCapture()
        with patch("agent.runtime.time.sleep"):
            AgentRuntime(client, capture).run()
        self.assertEqual(len(client.cursor), 3)
        self.assertNotIn("start", capture.calls)

    def test_authentication_failure_stops_cleanly(self):
        capture = FakeCapture()
        client = FakeClient([BridgeError("HTTP 401", status=401)])
        with self.assertRaises(BridgeError):
            AgentRuntime(client, capture).run()
        self.assertEqual(capture.calls, ["shutdown"])

    def test_prolonged_network_loss_stops_capture(self):
        capture = FakeCapture()
        capture.running = True
        client = FakeClient([BridgeError("unavailable"), KeyboardInterrupt])
        runtime = AgentRuntime(client, capture)
        runtime.last_success = 0
        with patch("agent.runtime.time.monotonic", return_value=40), patch("agent.runtime.time.sleep"):
            runtime.run()
        self.assertIn("stop", capture.calls)
        self.assertIn("shutdown", capture.calls)


if __name__ == "__main__":
    unittest.main()

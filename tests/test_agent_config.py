"""Verify stable instance credentials and HTTPS configuration."""
import os
import unittest
from unittest.mock import patch

from agent.config import AgentConfig


class AgentConfigTests(unittest.TestCase):
    def valid(self, **kwargs):
        env = {
            "ALEXIOS_BRIDGE_URL": "https://alexios-web.vercel.app",
            "ALEXIOS_AGENT_TOKEN": "a" * 32,
            "ALEXIOS_INSTANCE_ID": "alx_" + "f" * 48,
        }
        env.update(kwargs)
        return env

    def test_production_configuration(self):
        with patch.dict(os.environ, self.valid(), clear=True):
            conf = AgentConfig.from_env()
        self.assertEqual(conf.bridge_url, "https://alexios-web.vercel.app")
        self.assertEqual(conf.instance_id, "alx_" + "f" * 48)

    def test_identifier_required_and_fixed(self):
        with patch.dict(os.environ, self.valid(ALEXIOS_INSTANCE_ID=""), clear=True):
            with self.assertRaisesRegex(ValueError, "ALEXIOS_INSTANCE_ID"):
                AgentConfig.from_env()
        with patch.dict(os.environ, self.valid(ALEXIOS_INSTANCE_ID="alexios-v1"), clear=True):
            with self.assertRaisesRegex(ValueError, "ALEXIOS_INSTANCE_ID"):
                AgentConfig.from_env()

    def test_insecure_remote_http_rejected(self):
        with patch.dict(os.environ, self.valid(
            ALEXIOS_BRIDGE_URL="http://alexios-web.vercel.app"
        ), clear=True):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                AgentConfig.from_env()

    def test_local_dev_http_permitted(self):
        with patch.dict(os.environ, self.valid(
            ALEXIOS_BRIDGE_URL="http://127.0.0.1:3000"
        ), clear=True):
            conf = AgentConfig.from_env()
        self.assertEqual(conf.bridge_url, "http://127.0.0.1:3000")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

from mydailynews.ai.factory import create_ai_client
from mydailynews.ai.llama_cpp_server_client import LlamaCppServerClient
from mydailynews.ai.base import JSONSchemaSpec
from mydailynews.app.config import _apply_environment_overrides, _validate_ai_runtime, load_config
from mydailynews.app.models import AIConfig, AppConfig


class TestAIEndpointFlexibility(TestCase):
    def test_ai_config_defaults_and_fields(self) -> None:
        cfg = AIConfig()
        self.assertEqual(cfg.backend, "llama_cpp_server")
        self.assertEqual(cfg.api_key, "")
        self.assertEqual(cfg.headers, {})

        custom = AIConfig(
            backend="openai_compatible",
            api_key="sk-test-key",
            headers={"HTTP-Referer": "https://example.com"},
        )
        self.assertEqual(custom.backend, "openai_compatible")
        self.assertEqual(custom.api_key, "sk-test-key")
        self.assertEqual(custom.headers["HTTP-Referer"], "https://example.com")

    def test_factory_creates_client_for_supported_backends(self) -> None:
        cfg_llama = AIConfig(backend="llama_cpp_server")
        client_llama = create_ai_client(cfg_llama)
        self.assertIsInstance(client_llama, LlamaCppServerClient)

        cfg_openai = AIConfig(backend="openai_compatible")
        client_openai = create_ai_client(cfg_openai)
        self.assertIsInstance(client_openai, LlamaCppServerClient)

        cfg_alias = AIConfig(backend="openai")
        client_alias = create_ai_client(cfg_alias)
        self.assertIsInstance(client_alias, LlamaCppServerClient)

        cfg_invalid = AIConfig(backend="unsupported_backend")
        with self.assertRaisesRegex(ValueError, "Unsupported ai backend"):
            create_ai_client(cfg_invalid)

    def test_post_chat_completion_passes_auth_and_custom_headers(self) -> None:
        cfg = AIConfig(
            backend="openai_compatible",
            base_url="https://api.openai.com/v1",
            api_key="sk-secret-token",
            headers={"X-Title": "MyNewsApp"},
        )
        client = LlamaCppServerClient(cfg)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}]
        }

        with patch("mydailynews.ai.llama_cpp_server_client.requests.post", return_value=mock_resp) as mock_post:
            res, finish_reason, _, _ = client._post_chat_completion({"model": "gpt-4o-mini", "messages": []})
            self.assertEqual(res, '{"ok": true}')
            self.assertEqual(finish_reason, "stop")

            self.assertTrue(mock_post.called)
            _, kwargs = mock_post.call_args
            headers = kwargs.get("headers", {})
            self.assertEqual(headers.get("Authorization"), "Bearer sk-secret-token")
            self.assertEqual(headers.get("X-Title"), "MyNewsApp")
            self.assertEqual(headers.get("Content-Type"), "application/json")

    def test_build_payload_omits_top_level_json_schema_for_openai_compatible(self) -> None:
        spec = JSONSchemaSpec(name="test_schema", schema={"type": "object"})

        cfg_openai = AIConfig(backend="openai_compatible")
        client_openai = LlamaCppServerClient(cfg_openai)
        payload_openai = client_openai._build_payload("sys", "usr", 256, json_schema=spec)
        self.assertIn("response_format", payload_openai)
        self.assertNotIn("json_schema", payload_openai)

        cfg_llama = AIConfig(backend="llama_cpp_server")
        client_llama = LlamaCppServerClient(cfg_llama)
        payload_llama = client_llama._build_payload("sys", "usr", 256, json_schema=spec)
        self.assertIn("response_format", payload_llama)
        self.assertIn("json_schema", payload_llama)

    def test_validate_ai_runtime_rejects_manage_server_for_openai_compatible(self) -> None:
        cfg = AIConfig(
            backend="openai_compatible",
            manage_server=True,
            server_executable="server.exe",
            server_model_path="model.gguf",
        )
        with self.assertRaisesRegex(ValueError, "manage_server=true is only supported for 'llama_cpp_server'"):
            _validate_ai_runtime(cfg, "ai_summary")

    def test_environment_variable_overrides(self) -> None:
        app_cfg = AppConfig(
            ai_summary=AIConfig(manage_server=True),
            ai_final=AIConfig(manage_server=True),
        )

        env_patch = {
            "MYDAILYNEWS_AI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_API_KEY": "sk-global-env-key",
            "MYDAILYNEWS_AI_BACKEND": "openai_compatible",
            "MYDAILYNEWS_AI_MODEL": "gpt-4o",
            "MYDAILYNEWS_AI_MANAGE_SERVER": "false",
        }
        with patch.dict(os.environ, env_patch, clear=False):
            _apply_environment_overrides(app_cfg)

        self.assertEqual(app_cfg.ai_summary.base_url, "https://api.openai.com/v1")
        self.assertEqual(app_cfg.ai_final.base_url, "https://api.openai.com/v1")
        self.assertEqual(app_cfg.ai_summary.api_key, "sk-global-env-key")
        self.assertEqual(app_cfg.ai_final.api_key, "sk-global-env-key")
        self.assertEqual(app_cfg.ai_summary.backend, "openai_compatible")
        self.assertEqual(app_cfg.ai_final.backend, "openai_compatible")
        self.assertEqual(app_cfg.ai_summary.server_model, "gpt-4o")
        self.assertEqual(app_cfg.ai_final.server_model, "gpt-4o")
        self.assertFalse(app_cfg.ai_summary.manage_server)
        self.assertFalse(app_cfg.ai_final.manage_server)

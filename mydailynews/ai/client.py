from __future__ import annotations

from typing import Any, Dict, Optional

from ollama import Client

from ..models import AIConfig
from ..utils import safe_json_load


class LocalAIClient:
    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self.client = Client(host=config.host, timeout=config.timeout_seconds)

    def complete_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
                options={"temperature": self.config.temperature},
            )
            content = response.message.content if hasattr(response, "message") else response["message"]["content"]
            return safe_json_load(content or "")
        except Exception:
            return None

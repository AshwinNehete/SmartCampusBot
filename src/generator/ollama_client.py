import logging
from typing import List, Optional, Any

import ollama


class OllamaClient:
    """Client for interacting with Ollama models"""

    def __init__(
        self, base_url: str = "http://localhost:11434", model: str = "llama2:7b-chat"
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client: Optional[ollama.Client] = None
        self._initialize_client()

    # ---------- internals ----------

    def _initialize_client(self):
        """Initialize Ollama client and validate connectivity."""
        try:
            self.client = ollama.Client(host=self.base_url)
            names = self._list_model_names()
            logging.info(f"Connected to Ollama at {self.base_url}")
            logging.info(f"Available models: {names or '(none)'}")
        except Exception as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            logging.error("Make sure Ollama is running and accessible")
            raise

    def _extract_name(self, item: Any) -> Optional[str]:
        """Extract a model name reliably from various item shapes."""
        # Typed object (pydantic/dataclass-like)
        if hasattr(item, "name") and isinstance(getattr(item, "name"), str):
            return getattr(item, "name")
        if hasattr(item, "model") and isinstance(getattr(item, "model"), str):
            return getattr(item, "model")

        # Mapping/dict-like
        if isinstance(item, dict):
            if isinstance(item.get("name"), str):
                return item["name"]
            if isinstance(item.get("model"), str):
                return item["model"]

        return None

    def _list_model_names(self) -> List[str]:
        """
        Return model names from the Python client, tolerating schema differences:
        - Newer clients: ListResponse with .models -> list[Model]
        - Older clients: {"models": [ ...dicts... ]}
        - Some variants: direct list
        """
        if not self.client:
            return []

        resp = self.client.list()

        # Case A: typed response with `.models`
        if hasattr(resp, "models"):
            items = getattr(resp, "models")
        # Case B: dict with "models"
        elif isinstance(resp, dict) and "models" in resp:
            items = resp["models"]
        # Case C: direct list
        elif isinstance(resp, list):
            items = resp
        else:
            raise ValueError(f"Unexpected list() schema: {type(resp).__name__}")

        if not isinstance(items, list):
            raise ValueError(
                f"Unexpected 'models' container type: {type(items).__name__}"
            )

        names: List[str] = []
        for m in items:
            n = self._extract_name(m)
            if n:
                names.append(n)
        return names

    def _normalized_targets(self, model_name: Optional[str]) -> List[str]:
        """Normalize acceptable tags: 'foo' ~ 'foo:latest'."""
        t = (model_name or self.model).strip()
        if ":" in t:
            base = t.split(":", 1)[0]
            return [t, base]
        else:
            return [t, f"{t}:latest"]

    # ---------- public API ----------

    def generate_response(
        self, prompt: str, max_tokens: int = 500, temperature: float = 0.7
    ) -> str:
        """Generate response using Ollama model."""
        if not self.client:
            raise RuntimeError("Ollama client not initialized")

        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            )

            # Newer clients return a typed object with .message.content
            if hasattr(response, "message") and hasattr(response.message, "content"):
                content = response.message.content
                if isinstance(content, str):
                    return content.strip()

            # Older clients may return a dict
            if isinstance(response, dict):
                msg = (
                    response.get("message")
                    if isinstance(response.get("message"), dict)
                    else None
                )
                content = (msg or {}).get("content")
                if isinstance(content, str):
                    return content.strip()

            logging.error(f"Unexpected response format from Ollama: {response!r}")
            return "I encountered an error generating a response."

        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return (
                "I'm having trouble generating a response right now. Please try again."
            )

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific model is available (robust to schema/tag differences)."""
        targets = self._normalized_targets(model_name)
        try:
            available = set(self._list_model_names())
            return any(t in available for t in targets)
        except Exception as e:
            logging.error(f"Error checking model availability: {e}")
            return False

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """Pull/download a model if not available."""
        target = (model_name or self.model).strip()
        try:
            logging.info(f"Pulling model: {target}")
            self.client.pull(target)
            logging.info(f"Successfully pulled model: {target}")
            return True
        except Exception as e:
            logging.error(f"Error pulling model {target}: {e}")
            return False

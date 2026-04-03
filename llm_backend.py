"""
llm_backend.py — Pluggable LLM abstraction for AgentOS

Priority order:
  1. Anthropic Claude (if ANTHROPIC_API_KEY is set)
  2. Local Ollama   (if Ollama is running locally)
  3. OpenAI         (if OPENAI_API_KEY is set)

All backends expose the same interface: llm.invoke(messages) -> str
"""

import os
import json
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 2048
    backend: str = "auto"          # "claude" | "ollama" | "openai" | "auto"
    #ollama_host: str = "http://localhost:11434"
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = "llama3.2"
    openai_model: str = "gpt-4o-mini"


class LLMMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class LLMResponse:
    def __init__(self, content: str, backend_used: str, model: str):
        self.content = content
        self.backend_used = backend_used
        self.model = model

    def __str__(self):
        return self.content


# ─── Backend implementations ────────────────────────────────────────────────

class ClaudeBackend:
    """Anthropic Claude via official SDK."""

    def __init__(self, config: LLMConfig):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            self.model = config.model or "claude-3-5-haiku-20241022"
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            logger.info(f"Claude backend initialised: {self.model}")
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        except KeyError:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")

    def invoke(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[m.to_dict() for m in messages],
        )
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        content = response.content[0].text
        return LLMResponse(content, backend_used="claude", model=self.model)


class OllamaBackend:
    """Local Ollama — zero cost, fully sovereign."""

    def __init__(self, config: LLMConfig):
        import urllib.request
        self.host = config.ollama_host
        self.model = config.ollama_model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._urllib = urllib.request
        # Verify Ollama is running
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=3)
            logger.info(f"Ollama backend initialised: {self.model} @ {self.host}")
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {self.host}: {e}")

    def invoke(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        import urllib.request, urllib.error

        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode()
        req = self._urllib.Request(
            f"{self.host}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with self._urllib.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        content = result["message"]["content"]
        return LLMResponse(content, backend_used="ollama", model=self.model)


class OpenAIBackend:
    """OpenAI as final fallback."""

    def __init__(self, config: LLMConfig):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            self.model = config.model or config.openai_model
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens
            logger.info(f"OpenAI backend initialised: {self.model}")
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")
        except KeyError:
            raise RuntimeError("OPENAI_API_KEY not set.")

    def invoke(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend([m.to_dict() for m in messages])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return LLMResponse(content, backend_used="openai", model=self.model)


# ─── Auto-selecting factory ──────────────────────────────────────────────────

class LLMBackend:
    """
    Automatically selects the best available backend.

    Usage:
        llm = LLMBackend()
        response = llm.invoke([LLMMessage("user", "Hello!")])
        print(response.content)
        print(response.backend_used)   # "claude" | "ollama" | "openai"
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._backend = self._init_backend()

    def _init_backend(self):
        order = self._resolve_order()
        errors = []

        for backend_name in order:
            try:
                if backend_name == "claude":
                    return ClaudeBackend(self.config)
                elif backend_name == "ollama":
                    return OllamaBackend(self.config)
                elif backend_name == "openai":
                    return OpenAIBackend(self.config)
            except RuntimeError as e:
                errors.append(f"  {backend_name}: {e}")
                logger.debug(f"Backend {backend_name} unavailable: {e}")

        raise RuntimeError(
            "No LLM backend available. Options:\n"
            "  1. Set ANTHROPIC_API_KEY\n"
            "  2. Run Ollama locally: ollama pull llama3.2\n"
            "  3. Set OPENAI_API_KEY\n"
            f"Errors:\n" + "\n".join(errors)
        )

    def _resolve_order(self) -> list[str]:
        if self.config.backend != "auto":
            return [self.config.backend]

        order = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            order.append("claude")
        order.append("ollama")   # Always try Ollama (no key needed)
        if os.environ.get("OPENAI_API_KEY"):
            order.append("openai")
        return order

    def invoke(self, messages: list[LLMMessage], system: str = "") -> LLMResponse:
        return self._backend.invoke(messages, system=system)

    def invoke_text(self, prompt: str, system: str = "") -> str:
        """Convenience method — pass a plain string, get a plain string back."""
        resp = self.invoke([LLMMessage("user", prompt)], system=system)
        return resp.content

    @property
    def backend_name(self) -> str:
        return self._backend.__class__.__name__.replace("Backend", "").lower()


# ─── Quick smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLMBackend()
    print(f"Using backend: {llm.backend_name}")
    resp = llm.invoke_text("Say 'AgentOS online' and nothing else.")
    print(f"Response: {resp}")

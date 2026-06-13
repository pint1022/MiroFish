from app.config import Config
from app.services import oasis_profile_generator
from app.services import simulation_config_generator
from app.utils import llm_client


def _patch_openai_constructor(monkeypatch, module):
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    return calls


def test_llm_client_uses_configured_openai_timeout(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "test-model")
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", 12.5, raising=False)
    calls = _patch_openai_constructor(monkeypatch, llm_client)

    llm_client.LLMClient()

    assert calls[-1]["timeout"] == 12.5


def test_simulation_config_generator_uses_configured_openai_timeout(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "test-model")
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", 12.5, raising=False)
    calls = _patch_openai_constructor(monkeypatch, simulation_config_generator)

    simulation_config_generator.SimulationConfigGenerator()

    assert calls[-1]["timeout"] == 12.5


def test_oasis_profile_generator_uses_configured_openai_timeout(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(Config, "LLM_MODEL_NAME", "test-model")
    monkeypatch.setattr(Config, "LLM_TIMEOUT_SECONDS", 12.5, raising=False)
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    calls = _patch_openai_constructor(monkeypatch, oasis_profile_generator)

    oasis_profile_generator.OasisProfileGenerator()

    assert calls[-1]["timeout"] == 12.5

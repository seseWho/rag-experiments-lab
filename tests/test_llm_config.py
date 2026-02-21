from rag_core.llm_config import load_llm_config


def test_load_llm_config_reads_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    cfg = load_llm_config()

    assert cfg.provider
    assert cfg.model
    assert cfg.max_tokens > 0

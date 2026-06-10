"""Embedding provider configuration tests."""

from app.core.config import Settings
from app.core.errors import ConfigurationError
from app.core.llm_provider import AnswerRequest, OpenAIProvider


class FakeEmbeddingResponseItem:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embedding):
        self.data = [FakeEmbeddingResponseItem(embedding)]


class FakeEmbeddingsClient:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return FakeEmbeddingResponse([0.1, 0.2, 0.3])


class FakeOpenAICompatibleClient:
    def __init__(self):
        self.embeddings = FakeEmbeddingsClient()


class FakeChatMessage:
    content = "Short answer:\nAihubmix answer."


class FakeChatChoice:
    message = FakeChatMessage()


class FakeChatResponse:
    choices = [FakeChatChoice()]


class FakeChatCompletionsClient:
    def __init__(self):
        self.last_request = None

    def create(self, **request):
        self.last_request = request
        return FakeChatResponse()


class FakeChatClient:
    def __init__(self):
        self.chat = type("ChatNamespace", (), {"completions": FakeChatCompletionsClient()})()


def test_dashscope_embedding_settings_strip_quotes_and_use_dashscope_model():
    """DashScope settings support the quoted values often copied into `.env`."""

    settings = Settings(
        _env_file=None,
        embed_model_type=' "dashscope"',
        embed_model_name=' "text-embedding-v3"',
        embed_api_key=' "test-key"',
        embed_base_url=' "https://dashscope.aliyuncs.com/compatible-mode/v1"',
    )

    assert settings.normalized_embed_model_type == "dashscope"
    assert settings.active_embedding_model == "text-embedding-v3"
    assert settings.active_embedding_api_key == "test-key"
    assert settings.active_embedding_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.active_embedding_dimensions is None


def test_blank_embed_dimensions_means_provider_default():
    """A blank EMBED_DIMENSIONS value should not break settings loading."""

    settings = Settings(_env_file=None, embed_model_type="dashscope", embed_dimensions="")

    assert settings.active_embedding_dimensions is None


def test_dashscope_embedding_batch_size_is_capped_at_ten():
    """DashScope rejects embedding batches larger than 10 input texts."""

    settings = Settings(_env_file=None, embed_model_type="dashscope", embedding_batch_size=16)

    assert settings.active_embedding_batch_size == 10


def test_openai_embedding_batch_size_uses_configured_value():
    """OpenAI-compatible defaults outside DashScope are not capped by this app."""

    settings = Settings(_env_file=None, embed_model_type="openai", embedding_batch_size=16)

    assert settings.active_embedding_batch_size == 16


def test_dashscope_embedding_request_uses_openai_compatible_client_without_dimensions():
    """DashScope embedding requests use EMBED_* settings and omit dimensions by default."""

    settings = Settings(
        _env_file=None,
        embed_model_type="dashscope",
        embed_model_name="text-embedding-v3",
        embed_api_key="test-key",
        embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    fake_client = FakeOpenAICompatibleClient()
    provider = OpenAIProvider(settings)
    provider._embedding_client = fake_client

    embeddings = provider.embed_texts(["hello"])

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert fake_client.embeddings.last_request == {
        "model": "text-embedding-v3",
        "input": ["hello"],
    }


def test_dashscope_embedding_requires_embed_api_key():
    """DashScope reports EMBED_API_KEY instead of OPENAI_API_KEY when missing."""

    provider = OpenAIProvider(
        Settings(
            _env_file=None,
            embed_model_type="dashscope",
            embed_model_name="text-embedding-v3",
            embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )

    try:
        provider.embed_texts(["hello"])
    except ConfigurationError as exc:
        assert "EMBED_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ConfigurationError")


def test_aihubmix_chat_settings_take_priority_over_openai_settings():
    """LLM_* settings drive answer generation while EMBED_* remains separate."""

    settings = Settings(
        _env_file=None,
        openai_api_key="openai-key",
        openai_chat_model="openai-chat-model",
        llm_api_key=' "aihubmix-key"',
        llm_model_id=' "aihubmix-model"',
        llm_base_url=' "https://aihubmix.com/v1"',
        embed_model_type="dashscope",
        embed_model_name="text-embedding-v3",
        embed_api_key="dashscope-key",
        embed_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    assert settings.normalized_llm_model_type == "aihubmix"
    assert settings.active_chat_model == "aihubmix-model"
    assert settings.active_chat_api_key == "aihubmix-key"
    assert settings.active_chat_base_url == "https://aihubmix.com/v1"
    assert settings.normalized_embed_model_type == "dashscope"


def test_aihubmix_answer_generation_uses_chat_completions():
    """Aihubmix is OpenAI-compatible and should use chat.completions."""

    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeChatClient()
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    answer = provider.generate_answer(
        AnswerRequest(
            question="How much protein?",
            evidence_context="Approved evidence",
            safety_mode="normal",
            safety_flags=[],
            max_tokens=123,
        )
    )

    assert answer == "Short answer:\nAihubmix answer."
    assert fake_client.chat.completions.last_request["model"] == "aihubmix-model"
    assert fake_client.chat.completions.last_request["max_tokens"] == 123
    assert "Approved evidence" in fake_client.chat.completions.last_request["messages"][0]["content"]

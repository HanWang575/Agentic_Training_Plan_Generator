"""Embedding provider configuration tests."""

from app.core.config import Settings
from app.core.errors import ConfigurationError
import pytest

from app.core.llm_provider import (
    AnswerRequest,
    OpenAIProvider,
    ProgramEvidenceQueryPlanRequest,
    ProgramInputExtractionRequest,
    ProgramPlanUpdateRequest,
    ProgramSOPStepDecisionRequest,
    StructuredOutputError,
)


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


class FakeSequenceChatMessage:
    def __init__(self, content):
        self.content = content


class FakeSequenceChatChoice:
    def __init__(self, content):
        self.message = FakeSequenceChatMessage(content)


class FakeSequenceChatResponse:
    def __init__(self, content):
        self.choices = [FakeSequenceChatChoice(content)]


class FakeSequenceChatCompletionsClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return FakeSequenceChatResponse(self.responses.pop(0))


class FakeSequenceChatClient:
    def __init__(self, responses):
        self.chat = type("ChatNamespace", (), {"completions": FakeSequenceChatCompletionsClient(responses)})()


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


def test_program_plan_update_repairs_malformed_json_once():
    """Large program JSON occasionally misses a comma; the provider should ask for repair once."""

    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeSequenceChatClient(
        [
            '{"title":"Broken Plan" "sessions":[]}',
            '{"title":"Fixed Plan","sessions":[]}',
        ]
    )
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    plan = provider.update_program_plan(
        ProgramPlanUpdateRequest(
            sop_step="programming_intensity_and_volume",
            decision={"summary": "Use conservative loading."},
            raw_plan={"title": "Draft Plan", "sessions": []},
            collected_inputs={},
            previous_decisions={},
        )
    )

    assert plan == {"title": "Fixed Plan", "sessions": []}
    assert len(fake_client.chat.completions.requests) == 2
    assert "Repair the following malformed JSON" in fake_client.chat.completions.requests[1]["messages"][0]["content"]


def test_structured_output_error_keeps_malformed_json_excerpts():
    """Failed repair should preserve compact raw and repair excerpts for trace debugging."""

    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeSequenceChatClient(
        [
            '{"step":"programming_intensity_and_volume" "decision":{}}',
            '{"step":"programming_intensity_and_volume" "decision":{}}',
        ]
    )
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    with pytest.raises(StructuredOutputError) as exc_info:
        provider.update_program_plan(
            ProgramPlanUpdateRequest(
                sop_step="programming_intensity_and_volume",
                decision={"summary": "Use conservative loading."},
                raw_plan={"title": "Draft Plan", "sessions": []},
                collected_inputs={},
                previous_decisions={},
            )
        )

    assert "after one repair attempt" in str(exc_info.value)
    assert '"step":"programming_intensity_and_volume"' in exc_info.value.raw_excerpt
    assert '"decision":{}' in exc_info.value.repair_excerpt


def test_program_sop_step_executor_has_room_for_long_intensity_decision():
    """The intensity SOP can cover many exercises, so its JSON budget should not be tiny."""

    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeSequenceChatClient(
        [
            (
                '{"step":"programming_intensity_and_volume",'
                '"decision":{"programming_intensity_and_volume":{}},'
                '"rationale":"ok","assumptions":[],"warnings":[]}'
            )
        ]
    )
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    provider.generate_program_sop_step_decision(
        ProgramSOPStepDecisionRequest(
            task_type="program_generation",
            sop_step="programming_intensity_and_volume",
            user_message="Create a program",
            collected_inputs={},
            memory_context="",
            weekly_summaries=[],
            evidence=[],
            previous_decisions={},
            raw_plan={"sessions": []},
        )
    )

    assert fake_client.chat.completions.requests[0]["max_tokens"] == 10000


def test_program_sop_step_repair_prompt_includes_error_and_expected_shape():
    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    malformed = (
        '{"step":"rest_periods","decision":{"rest_periods":{"Bench Press":150},'
        '"rationale":"nested rationale"}},"rationale":"top rationale","assumptions":[],"warnings":[]}'
    )
    repaired = (
        '{"step":"rest_periods","decision":{"rest_periods":{"Bench Press":150}},'
        '"rationale":"top rationale","assumptions":[],"warnings":[]}'
    )
    fake_client = FakeSequenceChatClient([malformed, repaired])
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    result = provider.generate_program_sop_step_decision(
        ProgramSOPStepDecisionRequest(
            task_type="program_generation",
            sop_step="rest_periods",
            user_message="Create a program",
            collected_inputs={},
            memory_context="",
            weekly_summaries=[],
            evidence=[],
            previous_decisions={},
            raw_plan={"sessions": []},
        )
    )

    repair_prompt = fake_client.chat.completions.requests[1]["messages"][0]["content"]
    assert result["rationale"] == "top rationale"
    assert "Parser error:" in repair_prompt
    assert "Extra data" in repair_prompt
    assert "Expected root shape:" in repair_prompt
    assert "The decision object must not contain rationale, assumptions, warnings, or step." in repair_prompt


def test_program_evidence_query_planner_returns_clean_queries():
    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeSequenceChatClient(
        [
            (
                '{"queries":['
                '{"query":"strength training sets reps RPE deload","purpose":"intensity evidence"},'
                '{"query":"strength training sets reps RPE deload","purpose":"duplicate"},'
                '{"query":"bad","purpose":"too short"}'
                '],"rationale":"step-specific"}'
            )
        ]
    )
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    result = provider.plan_program_evidence_queries(
        ProgramEvidenceQueryPlanRequest(
            sop_step="programming_intensity_and_volume",
            user_message="Create a strength plan.",
            collected_inputs={"goal": "strength"},
            relevant_previous_decisions={"periodization": {"weeks": 12}},
            attempt=1,
        )
    )

    assert result.queries == [
        {"query": "strength training sets reps RPE deload", "purpose": "intensity evidence"}
    ]
    assert result.rationale == "step-specific"
    assert fake_client.chat.completions.requests[0]["max_tokens"] == 800


def test_program_input_extractor_accepts_expanded_required_inputs():
    settings = Settings(
        _env_file=None,
        llm_api_key="aihubmix-key",
        llm_model_id="aihubmix-model",
        llm_base_url="https://aihubmix.com/v1",
    )
    fake_client = FakeSequenceChatClient(
        [
            (
                '{"inputs":{"goal":"strength","physiological_priorities":["strength","power"],'
                '"age":29,"sex":"male","height_cm":175,"bodyweight_kg":80,'
                '"medical_history":"none","baseline_assessments":{"bench_1rm_kg":115},'
                '"lifestyle_work_demands":"desk job","days_per_week":3,'
                '"training_days":["Monday","Wednesday","Friday"],"session_duration_minutes":60,'
                '"program_duration_weeks":8,"experience":"intermediate","equipment":"full gym",'
                '"screening":"clear"}}'
            )
        ]
    )
    provider = OpenAIProvider(settings)
    provider._chat_client = fake_client

    result = provider.extract_program_inputs(
        ProgramInputExtractionRequest(user_message="Create a plan")
    )

    assert result["physiological_priorities"] == ["strength", "power"]
    assert result["age"] == 29
    assert result["sex"] == "male"
    assert result["height_cm"] == 175
    assert result["bodyweight_kg"] == 80
    assert result["baseline_assessments"] == {"bench_1rm_kg": 115}
    assert result["training_days"] == ["monday", "wednesday", "friday"]
    assert result["program_duration_weeks"] == 8

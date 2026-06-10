"""Provider adapter used by embedding and answer-generation nodes."""

from collections.abc import Sequence
from dataclasses import dataclass

from app.core.config import Settings
from app.core.errors import ConfigurationError


@dataclass(slots=True)
class AnswerRequest:
    """Input shape for answer generation."""

    question: str
    evidence_context: str
    safety_mode: str
    safety_flags: list[dict]
    max_tokens: int


class OpenAIProvider:
    """Small adapter around OpenAI-compatible APIs.

    The rest of the code calls this adapter instead of importing the OpenAI SDK
    directly. That keeps graph nodes easy to test with fake providers.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._chat_client = None
        self._embedding_client = None

    def _require_chat_client(self):
        """Return the chat client or raise a clear configuration error."""

        provider = self.settings.normalized_llm_model_type
        if provider not in {"openai", "aihubmix"}:
            raise ConfigurationError(
                f"Unsupported LLM_MODEL_TYPE '{self.settings.llm_model_type}'. "
                "Supported values are 'openai' and 'aihubmix'."
            )
        api_key = self.settings.active_chat_api_key
        if not api_key:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "LLM_API_KEY"
            raise ConfigurationError(
                f"{key_name} is not configured. The app can start, but answer generation needs a chat API key."
            )
        if self._chat_client is None:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if provider == "aihubmix":
                if not self.settings.active_chat_base_url:
                    raise ConfigurationError(
                        "LLM_BASE_URL is not configured. aihubmix needs an OpenAI-compatible base URL."
                    )
                client_kwargs["base_url"] = self.settings.active_chat_base_url
            self._chat_client = OpenAI(**client_kwargs)
        return self._chat_client

    def _require_embedding_client(self):
        """Return an OpenAI-compatible embedding client.

        Supported provider values:
            `openai`: uses OPENAI_API_KEY and OpenAI's default base URL.
            `dashscope`: uses EMBED_API_KEY and EMBED_BASE_URL.
        """

        provider = self.settings.normalized_embed_model_type
        if provider not in {"openai", "dashscope"}:
            raise ConfigurationError(
                f"Unsupported EMBED_MODEL_TYPE '{self.settings.embed_model_type}'. "
                "Supported values are 'openai' and 'dashscope'."
            )
        api_key = self.settings.active_embedding_api_key
        if not api_key:
            key_name = "OPENAI_API_KEY" if provider == "openai" else "EMBED_API_KEY"
            raise ConfigurationError(
                f"{key_name} is not configured. Semantic indexing and retrieval need an embedding API key."
            )
        if self._embedding_client is None:
            from openai import OpenAI

            client_kwargs = {"api_key": api_key}
            if provider == "dashscope":
                if not self.settings.active_embedding_base_url:
                    raise ConfigurationError(
                        "EMBED_BASE_URL is not configured. DashScope compatible mode needs a base URL."
                    )
                client_kwargs["base_url"] = self.settings.active_embedding_base_url
            self._embedding_client = OpenAI(**client_kwargs)
        return self._embedding_client

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Create embeddings.

        Parameters:
            texts: sequence with shape `[text_count]`.
        Returns:
            list[list[float]]: shape `[text_count, embedding_dimensions]`.
        """

        client = self._require_embedding_client()
        request = {
            "model": self.settings.active_embedding_model,
            "input": list(texts),
        }
        if self.settings.active_embedding_dimensions is not None:
            request["dimensions"] = self.settings.active_embedding_dimensions
        response = client.embeddings.create(**request)
        return [item.embedding for item in response.data]

    def generate_answer(self, request: AnswerRequest) -> str:
        """Generate a grounded answer from the compact evidence context.

        Parameters:
            request: AnswerRequest with the user question and retrieved context.
        Returns:
            str: answer formatted with Short answer, Practical guidance,
            Evidence used, and Safety note sections.
        """

        prompt = (
            "You are the Evidence-Based Q&A Assistant for a personal trainer AI.\n"
            "Answer only from the evidence context. If the evidence is not enough, "
            "say it is insufficient. Do not invent citations.\n\n"
            f"Safety mode: {request.safety_mode}\n"
            f"Safety flags: {request.safety_flags}\n\n"
            f"Question:\n{request.question}\n\n"
            f"Evidence context:\n{request.evidence_context}\n\n"
            "Use this exact structure:\n"
            "Short answer:\n...\n\n"
            "Practical guidance:\n...\n\n"
            "Evidence used:\n- [source_id / chunk_id] title, page/section\n\n"
            "Safety note:\n..."
        )
        client = self._require_chat_client()
        if self.settings.normalized_llm_model_type == "openai" and not self.settings.active_chat_base_url:
            response = client.responses.create(
                model=self.settings.active_chat_model,
                input=prompt,
                max_output_tokens=request.max_tokens,
            )
            return response.output_text

        response = client.chat.completions.create(
            model=self.settings.active_chat_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=request.max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""

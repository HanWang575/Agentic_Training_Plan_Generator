"""Typed runtime settings for the evidence-based Q&A block.

Input shape:
    Environment variables matching the field names below.
Output shape:
    One Settings object used by API routes, services, UI, and MCP tools.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from `.env` and environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    app_env: str = "local"
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    streamlit_port: int = 8501

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_chat_model: str = "gpt-5.4-mini"
    openai_eval_model: str = "gpt-5.5"
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 1024

    llm_model_type: str | None = Field(default=None, validation_alias="LLM_MODEL_TYPE")
    llm_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_MODEL_ID", "OPENAI_CHAT_MODEL"),
    )
    llm_api_key: str | None = Field(default=None, validation_alias="LLM_API_KEY")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")

    embed_model_type: str = Field(default="openai", validation_alias="EMBED_MODEL_TYPE")
    embed_model_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBED_MODEL_NAME", "OPENAI_EMBEDDING_MODEL"),
    )
    embed_api_key: str | None = Field(default=None, validation_alias="EMBED_API_KEY")
    embed_base_url: str | None = Field(default=None, validation_alias="EMBED_BASE_URL")
    embed_dimensions: int | None = Field(default=None, validation_alias="EMBED_DIMENSIONS")

    @field_validator("embed_dimensions", mode="before")
    @classmethod
    def empty_embed_dimensions_to_none(cls, value):
        """Allow `EMBED_DIMENSIONS=` to mean provider default dimensions."""

        if value is None:
            return None
        if isinstance(value, str) and not value.strip().strip('"').strip("'"):
            return None
        return value

    sqlite_path: Path = Path("data/app.sqlite")
    chroma_path: Path = Path("data/chroma")
    chroma_collection: str = "qa_approved_chunks_v1"

    upload_dir: Path = Path("data/uploads")
    rag_corpus_dir: Path = Path("RAG_corpus")
    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 100
    keyword_top_k: int = 30
    semantic_top_k: int = 30
    final_context_top_k: int = 12
    max_context_tokens: int = 12000
    answer_max_tokens: int = 1000
    hybrid_rrf_k: int = 60
    embedding_batch_size: int = 10
    maximum_upload_bytes: int = 100 * 1024 * 1024

    @property
    def api_base_url(self) -> str:
        """Output shape: `http://host:port`, used by Streamlit and MCP clients."""

        return f"http://{self.api_host}:{self.api_port}"

    @property
    def normalized_embed_model_type(self) -> str:
        """Output shape: provider name normalized to lowercase, e.g. `openai`."""

        return self.embed_model_type.strip().strip('"').strip("'").lower()

    @property
    def normalized_llm_model_type(self) -> str:
        """Output shape: chat provider name, e.g. `aihubmix` or `openai`."""

        if self.llm_model_type:
            return self.llm_model_type.strip().strip('"').strip("'").lower()
        if self.llm_api_key or self.llm_base_url:
            return "aihubmix"
        return "openai"

    @property
    def active_chat_model(self) -> str:
        """Return the model ID used for answer generation."""

        if self.llm_model_id:
            return self.llm_model_id.strip().strip('"').strip("'")
        return self.openai_chat_model

    @property
    def active_chat_api_key(self) -> str | None:
        """Return the API key used only for chat/LLM calls."""

        if self.llm_api_key:
            return self.llm_api_key.strip().strip('"').strip("'")
        if self.normalized_llm_model_type == "openai":
            return self.openai_api_key
        return None

    @property
    def active_chat_base_url(self) -> str | None:
        """Return an OpenAI-compatible base URL for non-OpenAI chat providers."""

        if not self.llm_base_url:
            return None
        return self.llm_base_url.strip().strip('"').strip("'")

    @property
    def active_embedding_model(self) -> str:
        """Return the embedding model name for the configured provider."""

        if self.embed_model_name:
            return self.embed_model_name.strip().strip('"').strip("'")
        if self.normalized_embed_model_type == "dashscope":
            return "text-embedding-v3"
        return self.openai_embedding_model

    @property
    def active_embedding_api_key(self) -> str | None:
        """Return the API key used only for embedding calls."""

        if self.embed_api_key:
            return self.embed_api_key.strip().strip('"').strip("'")
        if self.normalized_embed_model_type == "openai":
            return self.openai_api_key
        return None

    @property
    def active_embedding_base_url(self) -> str | None:
        """Return the OpenAI-compatible base URL for non-OpenAI embeddings."""

        if not self.embed_base_url:
            return None
        return self.embed_base_url.strip().strip('"').strip("'")

    @property
    def active_embedding_dimensions(self) -> int | None:
        """Return dimensions to send with embedding calls, if supported."""

        if self.embed_dimensions is not None:
            return self.embed_dimensions
        if self.normalized_embed_model_type == "openai":
            return self.openai_embedding_dimensions
        return None

    @property
    def active_embedding_batch_size(self) -> int:
        """Return provider-safe embedding batch size.

        DashScope rejects embedding requests with more than 10 input texts, so
        cap the configured value for that provider. Other providers use the
        configured value unchanged, with a minimum of 1.
        """

        batch_size = max(1, self.embedding_batch_size)
        if self.normalized_embed_model_type == "dashscope":
            return min(batch_size, 10)
        return batch_size


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings object.

    Parameters:
        None.
    Returns:
        Settings: one process-wide configuration object.
    """

    return Settings()

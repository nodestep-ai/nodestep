from __future__ import annotations

from pydantic import Field

from nodestep.exceptions import IntegrationNotInstalledError

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError as error:
    raise IntegrationNotInstalledError("openai", "openai") from error


class OpenAISettings(BaseSettings):
    """OpenAI connection settings read from ``OPENAI_*`` variables; empty ones count as unset.

    The table in the [module introduction][nodestep.chat.integrations.openai]
    lists the variable and the default of each field.
    """

    api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
        description="OpenAI API key.",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
        description="Base URL for the OpenAI-compatible API; None means the OpenAI API.",
    )
    organization: str | None = Field(
        default=None,
        validation_alias="OPENAI_ORG_ID",
        description="OpenAI organization ID.",
    )
    project: str | None = Field(
        default=None,
        validation_alias="OPENAI_PROJECT_ID",
        description="OpenAI project ID.",
    )
    timeout: float = Field(
        default=60.0,
        validation_alias="OPENAI_TIMEOUT",
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=3,
        validation_alias="OPENAI_MAX_RETRIES",
        description="Maximum retry attempts for transient failures.",
    )

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=True,
        env_ignore_empty=True,
    )


__all__ = ["OpenAISettings"]

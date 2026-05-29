"""Application settings loaded from environment / .env file.

Single source of truth for all config. Import `settings` anywhere
you need a value rather than reading os.environ directly — this gives
you typing, validation, and a clear contract.
"""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings.

    Values are loaded in this priority order (highest wins):
      1. Environment variables
      2. .env file
      3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # Gemini
    gemini_api_key: SecretStr = Field(
        ...,
        description="API key for Gemini. Required.",
    )

    # LLM defaults — prefixed QS_ to avoid clashing with other tools' env vars
    model: str = Field(
        default="gemini-2.5-pro",
        alias="GEMINI_MODEL",
        description="Default model for LLM calls.",
    )
    max_tokens: int = Field(
        default=2048,
        alias="QS_MAX_TOKENS",
    )
    temperature: float = Field(
        default=1.0,
        alias="QS_TEMPERATURE",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        alias="QS_LOG_LEVEL",
    )


# Module-level singleton — import this directly: `from quickstart import settings`
settings = Settings()  # type: ignore[call-arg]

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "PaperPilot API"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite+aiosqlite:///./paperpilot.db"
    frontend_origin: str = "http://localhost:3000"

    llm_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str | None = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices(
            "LLM_BASE_URL",
            "DEEPSEEK_BASE_URL",
            "OPENAI_BASE_URL",
        ),
    )
    llm_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias=AliasChoices("LLM_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL"),
    )
    llm_temperature: float = Field(
        default=0.1,
        validation_alias=AliasChoices(
            "LLM_TEMPERATURE",
            "DEEPSEEK_TEMPERATURE",
            "OPENAI_TEMPERATURE",
        ),
    )

    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_timeout_seconds: float = 20.0
    max_tool_iterations: int = Field(default=3, ge=1, le=8)
    supervisor_max_steps: int = Field(default=8, ge=2, le=20)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

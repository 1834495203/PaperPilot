from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, computed_field
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

    openalex_api_url: str = "https://api.openalex.org"
    openalex_api_key: SecretStr = SecretStr("")
    openalex_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    openalex_min_request_interval_seconds: float = Field(default=0.1, ge=0, le=60)
    openalex_max_retries: int = Field(default=1, ge=0, le=3)
    openalex_retry_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    semantic_scholar_api_url: str = "https://api.semanticscholar.org/graph/v1"
    semantic_scholar_api_key: SecretStr = SecretStr("")
    semantic_scholar_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    semantic_scholar_min_request_interval_seconds: float = Field(default=1.0, ge=0, le=60)
    semantic_scholar_max_retries: int = Field(default=1, ge=0, le=3)
    semantic_scholar_retry_backoff_seconds: float = Field(default=1.0, ge=0, le=60)
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    arxiv_timeout_seconds: float = 20.0
    arxiv_min_request_interval_seconds: float = Field(default=3.0, ge=0, le=60)
    arxiv_max_retries: int = Field(default=1, ge=0, le=3)
    arxiv_retry_backoff_seconds: float = Field(default=3.0, ge=0, le=60)
    pdf_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    pdf_max_bytes: int = Field(default=20_000_000, ge=1_000_000, le=100_000_000)
    pdf_max_pages: int = Field(default=80, ge=1, le=500)
    reader_max_input_chars: int = Field(default=100_000, ge=10_000, le=500_000)
    reader_max_retrieval_rounds: int = Field(default=2, ge=1, le=5)
    search_max_iterations: int = Field(default=2, ge=1, le=3)
    max_tool_iterations: int = Field(default=3, ge=1, le=8)
    supervisor_max_steps: int = Field(default=8, ge=2, le=20)
    enable_research_planner: bool = True
    planner_max_answer_dimensions: int = Field(default=4, ge=1, le=6)

    embedding_api_key: SecretStr = Field(
        default=SecretStr("ollama"),
        validation_alias="EMBEDDING_API_KEY",
    )
    embedding_base_url: str | None = Field(
        default="http://localhost:11434/v1",
        validation_alias="EMBEDDING_BASE_URL",
    )
    embedding_model: str = "qwen3-embedding:latest"
    embedding_dimensions: int | None = Field(default=None, ge=1)
    vector_db_path: Path = Path("./data/chroma")
    vector_collection: str = "paperpilot_tree_chunks"
    # 索引指纹不匹配时拒绝读写，避免把不同 embedding 模型写进同一个向量空间
    vector_enforce_index_signature: bool = True
    paper_library_path: Path = Path("./data/papers")
    upload_max_bytes: int = Field(default=30_000_000, ge=1_000_000, le=100_000_000)
    tree_chunk_max_chars: int = Field(default=1_800, ge=200, le=20_000)
    tree_chunk_overlap_sentences: int = Field(default=1, ge=0, le=10)
    retrieval_initial_top_k: int = Field(default=12, ge=1, le=100)
    retrieval_final_top_k: int = Field(default=8, ge=1, le=50)
    retrieval_max_expanded_per_hit: int = Field(default=8, ge=1, le=100)
    retrieval_max_candidates: int = Field(default=40, ge=1, le=500)
    retrieval_max_chunks_per_paper: int = Field(default=3, ge=1, le=100)
    retrieval_paper_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_sections_per_paper: int = Field(default=3, ge=1, le=20)
    retrieval_global_fallback_top_k: int = Field(default=6, ge=1, le=100)
    retrieval_keyword_top_k: int = Field(default=10, ge=1, le=100)
    retrieval_keyword_enabled: bool = True
    retrieval_keyword_filter_limit: int = Field(default=400, ge=1, le=5_000)
    retrieval_rerank_candidate_limit: int = Field(default=30, ge=1, le=200)
    retrieval_multi_paper_chunks_per_paper: int = Field(default=2, ge=1, le=10)
    retrieval_survey_paper_top_k: int = Field(default=20, ge=1, le=100)
    retrieval_survey_final_top_k: int = Field(default=16, ge=1, le=64)
    retrieval_max_final_top_k: int = Field(default=24, ge=1, le=128)
    retrieval_max_expansion_levels: int = Field(default=2, ge=1, le=5)
    retrieval_rrf_constant: int = Field(default=60, ge=1, le=1_000)
    # 融合分数在初排中的加权增益（0 表示只做候选合并，不参与排序）
    retrieval_rrf_ranking_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    retrieval_min_ranking_score: float = Field(default=0.20, ge=-1.0, le=1.0)
    retrieval_score_window: float = Field(default=0.18, ge=0.0, le=2.0)
    retrieval_mmr_top_k: int = Field(default=30, ge=1, le=200)
    retrieval_mmr_lambda: float = Field(default=0.70, ge=0.0, le=1.0)
    reranker_model: str | None = None
    reranker_device: str | None = None
    reranker_max_length: int = Field(default=512, ge=32, le=8_192)

    enable_citation_verification: bool = True
    citation_verification_max_citations: int = Field(default=12, ge=1, le=40)
    run_replay_buffer_size: int = Field(default=2_000, ge=100, le=100_000)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def durable_paths(self) -> dict[str, str]:
        """Every location whose loss destroys research data.

        Deployments must mount all of them together; the health endpoint reports
        them so a restore can be verified instead of assumed.
        """

        return {
            "database": self.sqlite_path or self.database_url,
            "papers": str(self.paper_library_path.resolve()),
            "vectors": str(self.vector_db_path.resolve()),
            "vector_collection": self.vector_collection,
        }

    @property
    def sqlite_path(self) -> str | None:
        """Resolved SQLite file for sqlite URLs, otherwise None.

        SQLAlchemy spells an absolute path with four slashes and a relative one
        with three, so the remainder is unwrapped before resolving.
        """

        prefix = "sqlite+aiosqlite://"
        if not self.database_url.startswith(prefix):
            return None
        remainder = self.database_url[len(prefix) :]
        raw = remainder
        if remainder.startswith("//"):
            raw = remainder[1:]
        elif remainder.startswith("/"):
            raw = remainder[1:]
        return str(Path(raw).resolve()) if raw else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

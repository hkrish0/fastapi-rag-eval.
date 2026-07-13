from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"

    chunk_size: int = 800

    raw_docs_dir: str = "data/raw"
    chroma_persist_dir: str = "data/chroma"


@lru_cache
def get_settings() -> Settings:
    return Settings()

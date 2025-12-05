"""
Configuration module using Pydantic Settings.
Provides type-safe configuration with automatic environment variable loading.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_model_name: str
    azure_openai_deployment: str

    # Azure Search
    search_service_endpoint: str
    search_index_name: str
    knowledge_agent_name: str

    # Azure Storage
    artifacts_storage_account_url: str
    artifacts_storage_container: str
    samples_storage_container: str

    # Server
    host: str = "0.0.0.0"
    port: int = 5000

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars
    }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

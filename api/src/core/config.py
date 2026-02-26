"""
Configuration module: Take .env file and load it into the environment variables
"""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API
    APP_NAME: str
    DEBUG: bool
    API_VERSION: str

    # server
    HOST: str
    PORT: int

    # LLM Configuration
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL: str

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # Groq
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str 

    # Vector Database - Qdrant
    QDRANT_URL: str
    QDRANT_COLLECTION: str
    QDRANT_VECTOR_SIZE: int

    # Paths
    DOCUMENTS_PATH: str

    # Tavily Search
    TAVILY_API_KEY: str

    # RAG configuration
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    TOP_K_RESULTS: int

    # admin credentials
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ADMIN_USERNAME: str
    ADMIN_PASSWORD_HASH: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()

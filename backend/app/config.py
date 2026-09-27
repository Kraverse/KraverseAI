from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    top_k: int = 5
    database_path: str = "data/knowledge.db"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="KRAVERSE_", extra="ignore")


settings = Settings()

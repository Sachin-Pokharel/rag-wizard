from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    mongodb_uri: str
    qdrant_url: str
    langchain_api_key: str
    qdrant_api_key: str
    env: str = "dev"

    class Config:
        env_file = ".env"  # used only in dev
        extra = "forbid"

settings = Settings()

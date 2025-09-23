# app/config.py
from pydantic import BaseSettings, Field, AnyHttpUrl
from typing import Optional
import os

class Settings(BaseSettings):
    # Groq config
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_base_url: str = Field("https://api.groq.ai/v1", env="GROQ_BASE_URL")
    groq_model: str = Field("gpt-neo-3.9b", env="GROQ_MODEL")  # default, override in env

    # Chroma DB
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")

    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

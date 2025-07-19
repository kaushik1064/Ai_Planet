# Configuration settings
from pydantic_settings import BaseSettings
from typing import Optional
import os
from functools import lru_cache

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Math Routing Agent"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Groq API settings
    GROQ_API_KEY: str = "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"
    GROQ_MODEL: str = "llama3-8b-8192"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    
    # Vector Database settings
    QDRANT_CLOUD_URL: str = "https://379d9ce3-70cb-46c4-938c-2a9f2f39fa7c.eu-west-1-0.aws.cloud.qdrant.io"
    QDRANT_API_KEY: Optional[str] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.CDo5pEgbkWLAESD8nP8eKngTxTnKYwfaXJ32HfrzU0U"
    QDRANT_COLLECTION_NAME: str = "ai_planet"
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./math_agent.db"
    
    # Redis settings (for caching)
    REDIS_URL: str = "redis://localhost:6379"
    
    # MCP settings
    MCP_SERVER_URL: str = "http://localhost:8001"
    TAVILY_API_KEY: Optional[str] = "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"
    SERPER_API_KEY: Optional[str] = "cf56d6050f860e5d7cc80e5129be27df0e48cea6"
    
    # Knowledge base settings
    KB_CHUNK_SIZE: int = 1000
    KB_CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Guardrails settings
    MAX_INPUT_LENGTH: int = 2000
    MAX_OUTPUT_LENGTH: int = 5000
    ALLOWED_DOMAINS: list = ["mathematics", "algebra", "geometry", "calculus", "statistics"]
    
    # Feedback settings
    FEEDBACK_THRESHOLD: float = 0.7
    MIN_FEEDBACK_SAMPLES: int = 3
    
    # JEE Benchmark settings
    JEE_DATASET_PATH: str = "D:/ai_Planet/data/dataset.json"
    
    class Config:
        env_file = ".env.example"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
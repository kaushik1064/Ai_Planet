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
    GROQ_API_KEY: str
    GROQ_MODEL: str = "llama3-8b-8192"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    
    # Vector Database settings
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "math_problems"
    QDRANT_API_KEY: Optional[str] = None
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./math_agent.db"
    
    # Redis settings (for caching)
    REDIS_URL: str = "redis://localhost:6379"
    
    # MCP settings
    MCP_SERVER_URL: str = "http://localhost:8001"
    TAVILY_API_KEY: Optional[str] = None
    SERPER_API_KEY: Optional[str] = None
    
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
    JEE_DATASET_PATH: str = "data/jee_benchmark.json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()
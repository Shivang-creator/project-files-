import os
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Data Analyst Agent Settings for Schema-Driven Analysis"""
    
    # API Configuration
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # AI Configuration
    ai_provider: str = Field(default="google", env="AI_PROVIDER")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    ai_model: str = Field(default="gemini-1.5-flash", env="AI_MODEL")
    ai_max_tokens: int = Field(default=2000, env="AI_MAX_TOKENS")
    
    # Request Configuration
    request_timeout: int = Field(default=300, env="REQUEST_TIMEOUT")  # 5 minutes for data processing
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB for large datasets
    
    # Schema Extraction Configuration
    max_schema_rows: int = Field(default=5, env="MAX_SCHEMA_ROWS")  # Sample rows for schema
    schema_timeout: int = Field(default=30, env="SCHEMA_TIMEOUT")  # Schema extraction timeout
    
    # Visualization Configuration
    max_image_size: int = Field(default=100000, env="MAX_IMAGE_SIZE")  # 100KB plot limit
    plot_dpi: int = Field(default=100, env="PLOT_DPI")  # Lower DPI for smaller files
    plot_width: int = Field(default=8, env="PLOT_WIDTH")
    plot_height: int = Field(default=6, env="PLOT_HEIGHT")
    
    # Analysis Configuration
    enable_regression: bool = Field(default=True, env="ENABLE_REGRESSION")
    correlation_threshold: float = Field(default=0.1, env="CORRELATION_THRESHOLD")
    max_rows: int = Field(default=10000, env="MAX_ROWS")  # Max rows to process for performance
    
    # Web Scraping Configuration
    user_agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        env="USER_AGENT"
    )
    max_retries: int = Field(default=3, env="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="RETRY_DELAY")  # seconds between retries
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    enable_detailed_logging: bool = Field(default=False, env="ENABLE_DETAILED_LOGGING")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False
    }


# Global settings instance
try:
    settings = Settings()
    print(f"✓ Settings loaded successfully")
    print(f"✓ AI Provider: {settings.ai_provider}")
    print(f"✓ Debug Mode: {settings.debug}")
except Exception as e:
    print(f"⚠ Warning: Could not load settings: {e}")
    print("Using default settings. Please check your .env file.")
    settings = Settings()

"""
Configuration and environment setup for MCP + LangChain project.

This module handles:
- Environment variable loading
- Logging configuration
- API key validation
- Core settings
"""

import os
import sys
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file (if exists)
load_dotenv()


class Config:
    """Central configuration management."""
    
    # API Keys - loaded from environment
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Project paths
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    DOCUMENTS_DIR: str = PROJECT_ROOT
    
    # LLM Configuration
    LLM_MODEL: str = "gpt-4o-mini"  # Cost-effective, reliable
    LLM_TEMPERATURE: float = 0.0  # Deterministic for production
    LLM_MAX_TOKENS: int = 4096
    
    # MCP Configuration
    MCP_SERVER_NAME: str = "filesystem"
    MCP_ALLOWED_DIRECTORY: str = PROJECT_ROOT
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration and fail fast if missing."""
        errors = []
        
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set. Set it as environment variable.")
        
        if errors:
            for error in errors:
                logging.error(error)
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        logging.info("✓ Configuration validated successfully")
    
    @classmethod
    def display(cls) -> None:
        """Display non-sensitive configuration for debugging."""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Project Root:     {cls.PROJECT_ROOT}")
        print(f"Documents Dir:    {cls.DOCUMENTS_DIR}")
        print(f"LLM Model:        {cls.LLM_MODEL}")
        print(f"LLM Temperature:  {cls.LLM_TEMPERATURE}")
        print(f"OpenAI API Key:   {'✓ Set' if cls.OPENAI_API_KEY else '✗ Not Set'}")
        print("="*60 + "\n")


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def validate_api_connection() -> bool:
    """
    Test OpenAI API connectivity.
    
    Returns:
        True if connection successful, raises exception otherwise
    """
    from openai import OpenAI
    
    logger = logging.getLogger(__name__)
    
    try:
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        
        # Simple test: list models (lightweight operation)
        logger.info("Testing OpenAI API connection...")
        models = client.models.list()
        
        # Verify we got a response
        model_count = len(list(models.data))
        logger.info(f"✓ OpenAI API connection successful ({model_count} models available)")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ OpenAI API connection failed: {e}")
        raise


if __name__ == "__main__":
    """Test configuration and API connectivity."""
    
    # Setup logging
    setup_logging(Config.LOG_LEVEL)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting configuration validation...")
    
    # Display configuration
    Config.display()
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    # Test API connectivity
    try:
        validate_api_connection()
        logger.info("✓ All systems ready")
    except Exception as e:
        logger.error(f"API validation failed: {e}")
        sys.exit(1)

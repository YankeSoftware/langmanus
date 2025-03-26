import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file with expanded logging
try:
    # Try to load from .env file
    result = load_dotenv(verbose=True)
    if result:
        logging.info("Environment variables loaded from .env file")
    else:
        logging.warning("No .env file found or it couldn't be loaded")
except Exception as e:
    logging.error(f"Error loading .env file: {str(e)}")

# Configure logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Model Configuration
REASONING_MODEL = os.getenv("REASONING_MODEL", "gpt-4-turbo")
BASIC_MODEL = os.getenv("BASIC_MODEL", "gpt-3.5-turbo")
AZURE_API_BASE = os.getenv("AZURE_API_BASE", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")

# Azure specific deployment configurations
REASONING_AZURE_DEPLOYMENT = os.getenv("REASONING_AZURE_DEPLOYMENT", "")
BASIC_AZURE_DEPLOYMENT = os.getenv("BASIC_AZURE_DEPLOYMENT", "")
VL_AZURE_DEPLOYMENT = os.getenv("VL_AZURE_DEPLOYMENT", "")

# Local LLM Configuration
LLM_TYPE = os.getenv("LLM_TYPE", "openai")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4-turbo")

# Tool Configuration
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
BRAVE_MAX_RESULTS = int(os.getenv("BRAVE_MAX_RESULTS", "5"))
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# Log the state of important API keys with masked values for security
for key_name in ["BRAVE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]:
    key_value = os.getenv(key_name, "")
    if key_value:
        masked_key = key_value[:4] + "*" * (len(key_value) - 4) if len(key_value) > 4 else "****"
        logging.debug(f"{key_name} is set: {masked_key}")
    else:
        logging.debug(f"{key_name} is not set")

# Chrome instance configuration
CHROME_PATH = os.getenv("CHROME_PATH", "")

# Additional Tool Configuration
DuckDuckGo_SAFESEARCH = os.getenv("DuckDuckGo_SAFESEARCH", "moderate")

# Reasoning LLM configuration (for complex reasoning tasks)
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL")
REASONING_API_KEY = os.getenv("REASONING_API_KEY")

# Non-reasoning LLM configuration (for straightforward tasks)
BASIC_BASE_URL = os.getenv("BASIC_BASE_URL")
BASIC_API_KEY = os.getenv("BASIC_API_KEY")

# Azure OpenAI configuration
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

# Vision-language LLM configuration (for tasks requiring visual understanding)
VL_MODEL = os.getenv("VL_MODEL", "gpt-4o")
VL_BASE_URL = os.getenv("VL_BASE_URL")
VL_API_KEY = os.getenv("VL_API_KEY")

# Chrome Instance configuration
CHROME_INSTANCE_PATH = os.getenv("CHROME_INSTANCE_PATH")
CHROME_HEADLESS = os.getenv("CHROME_HEADLESS", "False") == "True"
CHROME_PROXY_SERVER = os.getenv("CHROME_PROXY_SERVER")
CHROME_PROXY_USERNAME = os.getenv("CHROME_PROXY_USERNAME")
CHROME_PROXY_PASSWORD = os.getenv("CHROME_PROXY_PASSWORD")

# Search tool configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")

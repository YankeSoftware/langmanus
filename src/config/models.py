import os
import logging
import json
from typing import Dict, Any, Optional, List, Union
import re

# Import LLM classes with proper error handling
try:
    from langchain_openai import ChatOpenAI, AzureChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI, AzureChatOpenAI

# Try to import DeepSeek from different possible locations
try:
    # First try the dedicated package
    from langchain_deepseek import ChatDeepSeek
    logger.debug("Successfully imported ChatDeepSeek from langchain_deepseek")
except ImportError:
    try:
        # Then try the community package
        from langchain_community.llms.deepseek import DeepSeek as ChatDeepSeek
        logger.debug("Successfully imported DeepSeek from langchain_community.llms.deepseek")
    except ImportError:
        logger.warning("Failed to import DeepSeek - creating placeholder class")
        # Create a placeholder class to avoid errors
        class ChatDeepSeek:
            """Placeholder for DeepSeek when the module is not available"""
            def __init__(self, *args, **kwargs):
                raise ImportError("The DeepSeek module could not be imported. Please install it with: pip install langchain-deepseek")

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    try:
        from langchain.chat_models import ChatAnthropic
    except ImportError:
        logger.warning("Failed to import ChatAnthropic")
        class ChatAnthropic:
            def __init__(self, *args, **kwargs):
                raise ImportError("The Anthropic module could not be imported")

from langchain_core.language_models import BaseChatModel
from langchain_core.callbacks import CallbackManager

logger = logging.getLogger(__name__)

# Default model configurations
DEFAULT_MODEL = "lm_studio"  # Default to LM Studio local model for regular tasks
COMPLEX_MODEL = "deepseek"  # Use DeepSeek for complex reasoning tasks

# Model temperature settings
DEFAULT_TEMPERATURE = 0.1
CREATIVE_TEMPERATURE = 0.2

# Configure model access
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# LM Studio configuration
LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.environ.get("LM_STUDIO_MODEL", "default")  # Will detect if not specified

class ModelManager:
    """
    Manages model selection and configuration for the agent system.
    Uses either LM Studio local models or DeepSeek for more complex tasks.
    """
    
    def __init__(self, api_config: Dict[str, Any] = None):
        """
        Initialize the model manager.
        
        Args:
            api_config: Optional configuration for API endpoints
        """
        self.api_config = api_config or {}
        
        # Get LM Studio config from api_config or environment
        self.lm_studio_base_url = self.api_config.get('lm_studio_base_url', LM_STUDIO_BASE_URL)
        self.lm_studio_model = self.api_config.get('lm_studio_model', LM_STUDIO_MODEL)
        
        # Check availability
        self.lm_studio_available = self._check_lm_studio()
        self.lm_studio_models = self._get_available_lm_studio_models()
        
        # Get DeepSeek config
        self.deepseek_api_key = self.api_config.get('deepseek_api_key', DEEPSEEK_API_KEY)
        self.deepseek_available = bool(self.deepseek_api_key)
        
        # Log model availability
        models_info = f"LM Studio: {self.lm_studio_available}"
        if self.lm_studio_available and self.lm_studio_models:
            models_info += f" (models: {', '.join(self.lm_studio_models)})"
        models_info += f", DeepSeek: {self.deepseek_available}"
        logger.info(f"Model availability - {models_info}")
    
    def _check_lm_studio(self) -> bool:
        """
        Check if LM Studio is available.
        
        Returns:
            Boolean indicating if LM Studio is available
        """
        try:
            import requests
            response = requests.get(f"{self.lm_studio_base_url}/models", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LM Studio check failed: {e}")
            return False
    
    def _get_available_lm_studio_models(self) -> List[str]:
        """
        Get list of available models from LM Studio.
        
        Returns:
            List of model IDs
        """
        if not self._check_lm_studio():
            return []
            
        try:
            import requests
            response = requests.get(f"{self.lm_studio_base_url}/models", timeout=3)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    return [model.get("id", "") for model in data["data"] if model.get("id")]
        except Exception as e:
            logger.warning(f"Failed to get LM Studio models: {e}")
        
        return []
    
    def get_model(self, model_type: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> BaseChatModel:
        """
        Get the appropriate model based on type and availability.
        
        Args:
            model_type: Type of model to use (lm_studio, deepseek)
            temperature: Temperature setting for the model
            
        Returns:
            An initialized language model
        """
        # Try to use the requested model type
        if model_type == "deepseek" and self.deepseek_available:
            return self._get_deepseek_model(temperature)
        elif model_type == "lm_studio" and self.lm_studio_available:
            return self._get_lm_studio_model(temperature)
        
        # Fall back based on availability
        if self.lm_studio_available:
            logger.info(f"Falling back to LM Studio (requested: {model_type})")
            return self._get_lm_studio_model(temperature)
        elif self.deepseek_available:
            logger.info(f"Falling back to DeepSeek (requested: {model_type})")
            return self._get_deepseek_model(temperature)
        else:
            logger.error("No models available! Using DeepSeek with default API.")
            return ChatDeepSeek(temperature=temperature)
    
    def _get_lm_studio_model(self, temperature: float) -> BaseChatModel:
        """
        Initialize and return a LM Studio model.
        
        Args:
            temperature: Temperature setting for the model
            
        Returns:
            Configured LM Studio model via OpenAI interface
        """
        from langchain_openai import ChatOpenAI
        
        logger.info(f"Using LM Studio model via {self.lm_studio_base_url}")
        
        # Use specified model ID if available, otherwise use default
        model_id = self.lm_studio_model
        if model_id == "default" and self.lm_studio_models:
            # Use first available model
            model_id = self.lm_studio_models[0]
            logger.info(f"Using default LM Studio model: {model_id}")
        
        return ChatOpenAI(
            model=model_id,
            openai_api_base=self.lm_studio_base_url,
            openai_api_key="sk-no-key-required",
            temperature=temperature,
        )
    
    def _get_deepseek_model(self, temperature: float) -> BaseChatModel:
        """
        Initialize and return a DeepSeek model.
        
        Args:
            temperature: Temperature setting for the model
            
        Returns:
            Configured DeepSeek model
        """
        logger.info(f"Using DeepSeek model")
        return ChatDeepSeek(
            api_key=self.deepseek_api_key,
            temperature=temperature,
        )

# Do NOT create global model manager instance here anymore
# Instead, create it when needed with the config

def get_model_for_agent(agent_type: str, api_config: Dict[str, Any] = None) -> BaseChatModel:
    """
    Get the appropriate model for a specific agent type.
    
    Args:
        agent_type: Type of agent (COORDINATOR, RESEARCHER, etc.)
        api_config: Optional configuration for API endpoints
        
    Returns:
        Configured language model for the agent
    """
    # Create a model manager with the provided config
    model_manager = ModelManager(api_config)
    
    # Use DeepSeek for complex reasoning tasks if available
    if agent_type in ["PLANNER", "CODER", "SUPERVISOR"] and model_manager.deepseek_available:
        return model_manager.get_model(COMPLEX_MODEL)
    
    # Use slightly higher temperature for creative tasks
    if agent_type in ["PLANNER", "CODER"]:
        return model_manager.get_model(DEFAULT_MODEL, CREATIVE_TEMPERATURE)
    
    # Default to LM Studio model with conservative temperature
    return model_manager.get_model(DEFAULT_MODEL) 
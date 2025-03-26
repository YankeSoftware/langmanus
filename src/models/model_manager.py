import os
import traceback
from log import logger

class ModelManager:
    """
    A class to manage LLM models and handle model selection and configuration.
    
    This helps centralize model configurations and provides fallback mechanisms.
    """
    
    def __init__(self):
        # Default API configurations
        self.default_api_config = {
            "model": os.environ.get("DEFAULT_MODEL", "mistral-7b-instruct-v0.3"),
            "temperature": float(os.environ.get("DEFAULT_TEMPERATURE", 0.7)),
            "max_tokens": int(os.environ.get("DEFAULT_MAX_TOKENS", 1500)),
            "top_p": float(os.environ.get("DEFAULT_TOP_P", 0.95)),
            "frequency_penalty": float(os.environ.get("DEFAULT_FREQUENCY_PENALTY", 0.0)),
            "presence_penalty": float(os.environ.get("DEFAULT_PRESENCE_PENALTY", 0.0)),
        }
        
        # Model-specific configurations
        self.model_configs = {
            "gpt-4": {
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 0.95,
            },
            "gpt-3.5-turbo": {
                "temperature": 0.8,
                "max_tokens": 2500,
                "top_p": 0.9,
            },
            "mistral-7b-instruct-v0.3": {
                "temperature": 0.7,
                "max_tokens": 1500,
                "top_p": 0.95,
            }
        }
        
        # Fallback models in order of preference
        self.fallback_models = [
            "mistral-7b-instruct-v0.3",
            "gpt-3.5-turbo",
            "gpt-4",
        ]
        
        # Load model availability from environment if provided
        self.available_models = os.environ.get("AVAILABLE_MODELS", "").split(",")
        if not self.available_models or self.available_models == [""]:
            self.available_models = list(self.model_configs.keys())
        
        logger.info(f"Initialized ModelManager with available models: {self.available_models}")
    
    def get_api_config(self, agent_type=None, custom_config=None):
        """
        Get API configuration for a specific agent type, with fallbacks.
        
        Args:
            agent_type (str, optional): Type of agent (RESEARCHER, PLANNER, etc.)
            custom_config (dict, optional): Custom configuration to override defaults
            
        Returns:
            dict: API configuration dictionary
        """
        try:
            # Start with default configuration
            api_config = self.default_api_config.copy()
            
            # Apply agent-specific configurations if available
            agent_config_key = f"{agent_type.upper()}_MODEL" if agent_type else None
            if agent_config_key and agent_config_key in os.environ:
                agent_model = os.environ[agent_config_key]
                if agent_model in self.model_configs:
                    api_config["model"] = agent_model
                    # Update with model-specific settings
                    api_config.update(self.model_configs[agent_model])
            
            # Apply custom configuration overrides
            if custom_config and isinstance(custom_config, dict):
                api_config.update(custom_config)
            
            # Ensure the selected model is available, or find a fallback
            if api_config["model"] not in self.available_models:
                logger.warning(f"Selected model {api_config['model']} is not available")
                for fallback in self.fallback_models:
                    if fallback in self.available_models:
                        logger.info(f"Using fallback model: {fallback}")
                        api_config["model"] = fallback
                        # Update with model-specific settings
                        api_config.update(self.model_configs.get(fallback, {}))
                        break
            
            # Validate numerical parameters
            for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                if param in api_config:
                    try:
                        if param == "max_tokens":
                            api_config[param] = int(api_config[param])
                        else:
                            api_config[param] = float(api_config[param])
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {param}, using default")
                        api_config[param] = self.default_api_config[param]
            
            return api_config
            
        except Exception as e:
            logger.error(f"Error in get_api_config: {str(e)}")
            logger.error(traceback.format_exc())
            # Return basic safe configuration in case of errors
            return self.default_api_config.copy()
    
    def is_model_available(self, model_name):
        """Check if a specific model is available"""
        return model_name in self.available_models 
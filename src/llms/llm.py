from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_deepseek import ChatDeepSeek
from src.llms.litellm_v2 import ChatLiteLLMV2 as ChatLiteLLM
from typing import Optional
from litellm import LlmProviders
import logging
import re
import os

from src.config import (
    REASONING_MODEL,
    REASONING_BASE_URL,
    REASONING_API_KEY,
    BASIC_MODEL,
    BASIC_BASE_URL,
    BASIC_API_KEY,
    VL_MODEL,
    VL_BASE_URL,
    VL_API_KEY,
    AZURE_API_BASE,
    AZURE_API_KEY,
    AZURE_API_VERSION,
    BASIC_AZURE_DEPLOYMENT,
    VL_AZURE_DEPLOYMENT,
    REASONING_AZURE_DEPLOYMENT,
)
from src.config.agents import LLMType

logger = logging.getLogger(__name__)

def is_lm_studio_url(base_url: Optional[str]) -> bool:
    """
    Check if the base URL is for LM Studio.
    LM Studio typically runs on localhost with port 1234 or has 'lmstudio' in the URL.
    """
    if not base_url:
        return False
    
    # Convert to lowercase for case-insensitive matching
    base_url_lower = base_url.lower()
    
    # Check for explicit lmstudio indicators
    if 'lmstudio' in base_url_lower:
        logger.info(f"Detected LM Studio in URL: {base_url}")
        return True
    
    # Check for localhost or 127.0.0.1 with typical LM Studio ports
    localhost_pattern = r'https?://(localhost|127\.0\.0\.1)(:\d+)?(/v1)?/?$'
    if re.match(localhost_pattern, base_url_lower):
        # Common ports used by LM Studio and similar local inference servers
        common_ports = ['1234', '8000', '5000', '3000', '8080']
        
        # If no port specified, assume it might be LM Studio
        if ':' not in base_url_lower or any(f":{port}" in base_url_lower for port in common_ports):
            logger.info(f"Detected potential local LM Studio server at {base_url}")
            return True
    
    # Check for environment variable that might indicate LM Studio usage
    if os.environ.get('USE_LM_STUDIO', '').lower() in ('true', '1', 'yes'):
        logger.info(f"LM Studio mode enabled by environment variable")
        return True
    
    return False

def create_openai_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatOpenAI:
    """
    Create a ChatOpenAI instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["base_url"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatOpenAI(**llm_kwargs)


def create_deepseek_llm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatDeepSeek:
    """
    Create a ChatDeepSeek instance with the specified configuration
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["api_base"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key

    return ChatDeepSeek(**llm_kwargs)


def create_azure_llm(
    azure_deployment: str,
    azure_endpoint: str,
    api_version: str,
    api_key: str,
    temperature: float = 0.0,
) -> AzureChatOpenAI:
    """
    create azure llm instance with specified configuration
    """
    return AzureChatOpenAI(
        azure_deployment=azure_deployment,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        api_key=api_key,
        temperature=temperature,
    )


def create_litellm(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs,
) -> ChatLiteLLM:
    """
    Create a ChatLiteLLM instance with the specified configuration.
    Will use the enhanced ChatLiteLLMV2 for LM Studio compatibility.
    """
    # Only include base_url in the arguments if it's not None or empty
    llm_kwargs = {"model": model, "temperature": temperature, **kwargs}

    if base_url:  # This will handle None or empty string
        llm_kwargs["api_base"] = base_url

    if api_key:  # This will handle None or empty string
        llm_kwargs["api_key"] = api_key
    
    # Detect LM Studio and use enhanced class
    if is_lm_studio_url(base_url):
        logger.info(f"Using enhanced ChatLiteLLMV2 for LM Studio with model {model}")
        return ChatLiteLLMV2(**llm_kwargs)
    
    # For other cases, also use our enhanced version as it's backward compatible
    return ChatLiteLLMV2(**llm_kwargs)


# Cache for LLM instances
_llm_cache: dict[LLMType, ChatOpenAI | ChatDeepSeek | AzureChatOpenAI | ChatLiteLLM] = (
    {}
)


def is_litellm_model(model_name: str) -> bool:
    """
    Check if the model name indicates it should be handled by LiteLLM.

    Args:
        model_name: The name of the model to check

    Returns:
        bool: True if the model should be handled by LiteLLM, False otherwise
    """
    return (
        model_name
        and "/" in model_name
        and model_name.split("/")[0] in [p.value for p in LlmProviders]
    )


def get_llm_by_type(
    llm_type: LLMType,
) -> ChatOpenAI | ChatDeepSeek | AzureChatOpenAI | ChatLiteLLM:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    if llm_type == "reasoning":
        if REASONING_AZURE_DEPLOYMENT:
            llm = create_azure_llm(
                azure_deployment=REASONING_AZURE_DEPLOYMENT,
                azure_endpoint=AZURE_API_BASE,
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY,
            )
        elif is_litellm_model(REASONING_MODEL):
            llm = create_litellm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
            )
        else:
            llm = create_deepseek_llm(
                model=REASONING_MODEL,
                base_url=REASONING_BASE_URL,
                api_key=REASONING_API_KEY,
            )
    elif llm_type == "basic":
        if BASIC_AZURE_DEPLOYMENT:
            llm = create_azure_llm(
                azure_deployment=BASIC_AZURE_DEPLOYMENT,
                azure_endpoint=AZURE_API_BASE,
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY,
            )
        elif is_litellm_model(BASIC_MODEL):
            llm = create_litellm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
            )
        else:
            llm = create_openai_llm(
                model=BASIC_MODEL,
                base_url=BASIC_BASE_URL,
                api_key=BASIC_API_KEY,
            )
    elif llm_type == "vision":
        if VL_AZURE_DEPLOYMENT:
            llm = create_azure_llm(
                azure_deployment=BASIC_AZURE_DEPLOYMENT,
                azure_endpoint=AZURE_API_BASE,
                api_version=AZURE_API_VERSION,
                api_key=AZURE_API_KEY,
            )
        elif is_litellm_model(VL_MODEL):
            llm = create_litellm(
                model=VL_MODEL,
                base_url=VL_BASE_URL,
                api_key=VL_API_KEY,
            )
        else:
            llm = create_openai_llm(
                model=VL_MODEL,
                base_url=VL_BASE_URL,
                api_key=VL_API_KEY,
            )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    _llm_cache[llm_type] = llm
    return llm


# Initialize LLMs for different purposes - now these will be cached
reasoning_llm = get_llm_by_type("reasoning")
basic_llm = get_llm_by_type("basic")
vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    # stream = reasoning_llm.stream("what is mcp?")
    # full_response = ""
    # for chunk in stream:
    #     full_response += chunk.content
    # print(full_response)

    print(basic_llm.invoke("Hello"))
    # print(vl_llm.invoke("Hello"))

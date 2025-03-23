import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def normalize_message(message: Any) -> Dict[str, Any]:
    """
    Convert any message format to a standard dictionary format.
    Handles different message formats such as:
    - Dictionaries with role/content
    - LangChain message objects with type/content
    
    Args:
        message: The message in any format
        
    Returns:
        A dictionary with standardized format: {"role": role, "content": content, ...}
    """
    # If already a dictionary with role and content
    if isinstance(message, dict) and "role" in message and "content" in message:
        return message
    
    # If it's a LangChain message object
    if hasattr(message, "type") and hasattr(message, "content"):
        # Convert from LangChain format
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            # Add other mappings as needed
        }
        role = role_map.get(message.type, "assistant")  # Default to assistant if unknown
        return {"role": role, "content": message.content}
    
    # If it's a simple string, treat as assistant message
    if isinstance(message, str):
        return {"role": "assistant", "content": message}
    
    # For unknown formats, log warning and return a default format
    logger.warning(f"Unknown message format: {type(message)}. Converting to default assistant message.")
    content = str(message) if message is not None else ""
    return {"role": "assistant", "content": content}


def normalize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize a list of messages to ensure consistent format.
    
    Args:
        messages: List of messages in potentially different formats
        
    Returns:
        List of normalized message dictionaries
    """
    return [normalize_message(msg) for msg in messages] 
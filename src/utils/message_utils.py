import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def extract_content(response: Any) -> str:
    """
    Extract string content from various response types.
    
    Args:
        response: The response from an LLM, which could be:
            - A string
            - An AIMessage object with content attribute
            - A dictionary with content key
            - Any other object that can be converted to string
            
    Returns:
        The extracted content as a string
    """
    # If it's a string with "content=" pattern from an AIMessage
    if isinstance(response, str) and "content=" in response:
        # Try to extract the actual content with regex
        import re
        match = re.search(r'content="(.*?)"', response)
        if match:
            return match.group(1)
    
    # If it has a content attribute (like AIMessage)
    if hasattr(response, "content"):
        content = response.content
        # If content is itself a complex object, recursively extract
        if not isinstance(content, str):
            return extract_content(content)
        return content
        
    # If it's a dict with content key
    elif isinstance(response, dict):
        if "content" in response:
            content = response["content"]
            # If content is itself a complex object, recursively extract
            if not isinstance(content, str):
                return extract_content(content)
            return content
            
        # Try to find any key that might contain text content
        for key in ["text", "message", "value", "result"]:
            if key in response and response[key]:
                return str(response[key])
                
        # Last resort: convert entire dict to string
        return str(response)
        
    # If it's a string, just return it
    elif isinstance(response, str):
        return response
        
    # For nested structures like lists
    elif isinstance(response, list) and response:
        # Try to extract content from the first element
        return extract_content(response[0])
        
    # Fallback: convert to string
    return str(response)

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

def clean_content_for_display(content: Any) -> str:
    """
    Clean message content for display to users, removing technical details
    and extracting the actual conversational content.
    
    Args:
        content: The content to clean, which may be in various formats:
            - Raw string that might include AIMessage representation
            - Message object 
            - String with agent tags
            
    Returns:
        A clean, human-readable string suitable for display
    """
    # First extract the core content
    if not isinstance(content, str):
        content = extract_content(content)
        
    # If we have a string, clean it up
    if isinstance(content, str):
        # 1. Handle AIMessage string representation with content=
        if "content=" in content and ("additional_kwargs" in content or "response_metadata" in content):
            import re
            match = re.search(r'content="(.*?)"', content)
            if match:
                content = match.group(1)
        
        # 2. Remove supervisor routing information
        if "Next:" in content and "Reasoning:" in content:
            # Extract only the reasoning part
            parts = content.split("Reasoning:", 1)
            if len(parts) > 1:
                content = parts[1].strip()
                
        # 3. Remove metadata about responses
        if "Responses:" in content:
            parts = content.split("Responses:", 1)
            if len(parts) > 1:
                # Remove everything after Responses:
                content = parts[0].strip()
        
        # 4. Clean up nested agent tags - look for [AGENT]: [AGENT]: pattern
        agent_tag_pattern = r'\[([\w]+)\]:'
        # First find all agent tags
        agent_tags = re.findall(agent_tag_pattern, content)
        
        # If we found multiple tags, clean them up
        if len(agent_tags) > 1:
            # Get the most recent agent tag
            latest_agent = agent_tags[-1]
            
            # Clean up nested tags - handle cases like [AGENT]: [AGENT]: content
            cleaned_content = ""
            # Split by agent tags
            parts = re.split(agent_tag_pattern, content)
            # The first part is before any agent tag, skip it if empty
            starting_index = 1 if parts[0].strip() == "" else 0
            
            # Rebuild the content with only the latest agent tag
            if starting_index == 0:
                cleaned_content = parts[0]
            
            # Add the last agent's content
            if len(parts) > starting_index + 1:
                final_content = parts[-1].strip()
                # Remove any remaining tags from the final content
                final_content = re.sub(agent_tag_pattern, '', final_content).strip()
                cleaned_content = f"[{latest_agent}]: {final_content}"
                content = cleaned_content
                
        # 5. Remove extra whitespace and newlines at beginning
        content = content.strip()
        
        # 6. Remove hashtags at the end
        if '#' in content:
            # Only remove hashtags if they're at the end of the message
            lines = content.split('\n')
            if any(line.strip().startswith('#') for line in lines[-2:]):
                # Remove the last lines if they start with hashtags
                while lines and lines[-1].strip().startswith('#'):
                    lines.pop()
                content = '\n'.join(lines)
    
    return content 
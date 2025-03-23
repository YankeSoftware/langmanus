import logging
import json
import json_repair
import re
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def repair_json_output(text: str) -> str:
    """
    Attempt to repair potentially malformed JSON in text.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        The repaired JSON string or the original if no repair was needed/possible
    """
    try:
        # Try to import json_repair for more robust repair
        from json_repair import repair_json
        return repair_json(text)
    except ImportError:
        logger.debug("json_repair module not available, using fallback method")
        
        try:
            # Check if we can parse it directly first
            json.loads(text)
            return text
        except json.JSONDecodeError:
            # Attempt some basic repairs for common issues
            
            # Replace single quotes with double quotes for keys and string values
            cleaned = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
            cleaned = re.sub(r':\s*\'([^\']*)\'', r': "\1"', cleaned)
            
            # Add quotes around unquoted keys
            cleaned = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cleaned)
            
            # Replace newlines in string values
            cleaned = re.sub(r'"\n\s*', r'" ', cleaned)
            
            # Remove trailing commas in objects and arrays
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            try:
                # Verify our repairs worked
                json.loads(cleaned)
                return cleaned
            except json.JSONDecodeError:
                # Return original if we couldn't repair it
                return text


def parse_structured_output(content: str, schema_class: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
    """
    Parse structured output from LLM, handling various formats.
    Specifically designed to handle outputs from Mistral and similar models
    that don't support OpenAI's function/tool calling natively.
    
    Args:
        content: The raw content from LLM
        schema_class: Optional Pydantic model class to validate against
        
    Returns:
        Parsed JSON object
    """
    if not content:
        logger.warning("Empty content received for structured output parsing")
        # Return minimal valid object based on schema if available
        if schema_class:
            fields = schema_class.__annotations__
            return {k: "" if v == str else False if v == bool else 0 for k, v in fields.items()}
        return {}
    
    # Try to extract JSON from the response
    json_str = ""
    
    # First try to extract from code blocks
    code_block_patterns = [
        r"```(?:json)?\s*([\s\S]*?)\s*```",  # Markdown code block with optional json
        r"`([\s\S]*?)`",  # Inline code block
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, content)
        if matches:
            # Use the longest match as it's most likely to be complete
            json_str = max(matches, key=len)
            logger.debug(f"Extracted JSON from code block: {json_str[:100]}...")
            break
    
    # If no code blocks found, look for JSON object patterns
    if not json_str:
        # Look for objects wrapped in curly braces
        matches = re.findall(r"(\{[\s\S]*?\})", content)
        if matches:
            # Use the longest match as it's most likely to be complete
            json_str = max(matches, key=len)
            logger.debug(f"Extracted JSON object: {json_str[:100]}...")
    
    # If still no JSON found, use the whole content
    if not json_str:
        json_str = content
        logger.debug("Using entire content as JSON")
    
    # Clean up the extracted content
    json_str = json_str.strip()
    
    # Try to parse the JSON with repair attempts
    try:
        result = json.loads(json_str)
        logger.debug("Successfully parsed JSON without repairs")
    except json.JSONDecodeError:
        try:
            # Attempt to repair JSON
            repaired = repair_json_output(json_str)
            logger.debug(f"Attempting to parse repaired JSON: {repaired[:100]}...")
            result = json.loads(repaired)
            logger.debug("Successfully parsed JSON after repairs")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON even after repairs: {e}")
            logger.error(f"Original content: {content[:200]}...")
            
            # Return minimal valid object based on schema if available
            if schema_class:
                required_fields = getattr(schema_class.__config__, "required_fields", 
                    [f for f in schema_class.__annotations__ if f not in schema_class.__fields_set__])
                
                result = {k: "" if isinstance(v, str) else False if isinstance(v, bool) else 0 
                          for k, v in schema_class.__annotations__.items() 
                          if k in required_fields}
            else:
                result = {}
    
    # Validate against schema if provided
    if schema_class:
        try:
            validated = schema_class(**result)
            logger.debug(f"Validated result against schema: {validated}")
            return validated.dict()
        except ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            # Return the unvalidated result as fallback
            return result
    
    return result

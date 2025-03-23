import os
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState
import logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render()
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def convert_message_to_dict(message: Any) -> Dict[str, Any]:
    """
    Convert different message types to a standardized dictionary format.
    Ensures compatibility with Mistral-7B-Instruct models.
    
    Args:
        message: A message object (either a dict or BaseMessage)
        
    Returns:
        A dictionary with 'role' and 'content' keys
    """
    # Handle None or empty messages
    if message is None:
        return {"role": "user", "content": ""}
        
    # Dictionary messages
    if isinstance(message, dict):
        # Copy the message to avoid modifying the original
        msg = message.copy()
        
        # Ensure the role is one of the supported types
        role = msg.get("role", "user")
        if role not in ["user", "assistant"]:
            # Convert any custom role to assistant with a prefix
            content = msg.get("content", "")
            if content:
                content = f"[{role.upper()}]: {content}"
            msg["role"] = "assistant"
            msg["content"] = content
        
        # Ensure content is a string
        if not isinstance(msg.get("content"), str):
            msg["content"] = str(msg.get("content", ""))
            
        return {
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        }
    
    # LangChain message types
    elif isinstance(message, BaseMessage):
        # Get the role, defaulting to user
        role = "user"
        if hasattr(message, "type"):
            role = message.type
            
        # Ensure role is compatible with Mistral
        if role not in ["user", "assistant"]:
            role = "assistant"
        
        # Format content based on message type/name
        content = message.content or ""
        if hasattr(message, "name") and message.name:
            content = f"[{message.name.upper()}]: {content}"
            
        return {
            "role": role,
            "content": content
        }
    
    # Fall back for any other object type
    else:
        logger.warning(f"Converting unknown message type to dictionary: {type(message)}")
        return {
            "role": "user",
            "content": str(message)
        }


def apply_prompt_template(prompt_name: str, state: Any) -> Dict[str, str]:
    """
    Apply template variables to a prompt template and return formatted message.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        A user message with the rendered template
    """
    # Log the inputs for debugging
    logger.debug(f"Template requested: {prompt_name}")
    logger.debug(f"State data type: {type(state)}")
    logger.debug(f"State contents: {state}")
    
    # Ensure state is a dictionary
    if not isinstance(state, dict):
        state = {"state": str(state)}
    
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
    }
    
    # Add state variables safely
    for key, value in state.items():
        state_vars[key] = value
    
    logger.debug(f"Template variables: {state_vars}")

    try:
        # Find template file
        template_path = f"{prompt_name}.md"
        template = None
        
        try:
            template = env.get_template(template_path)
            logger.debug(f"Template found: {template_path}")
        except Exception as template_error:
            logger.error(f"Error loading template {prompt_name}: {template_error}")
            # Create a fallback template string
            fallback_text = f"You are acting as the {prompt_name.capitalize()} agent.\n\n"
            fallback_text += f"Current time: {state_vars['CURRENT_TIME']}\n\n"
            
            if 'state' in state_vars:
                fallback_text += f"Current state:\n{state_vars['state']}\n\n"
                
            fallback_text += "Please help with the current task based on your role."
            
            # Return fallback message
            return {"role": "user", "content": fallback_text}
        
        # Render the template with variables
        try:
            system_prompt = template.render(**state_vars)
            logger.debug(f"Template rendered successfully")
        except Exception as render_error:
            logger.error(f"Error rendering template {prompt_name}: {render_error}")
            logger.error(f"Template variables: {state_vars}")
            # Return a simplified message
            return {"role": "user", "content": f"You are acting as the {prompt_name.capitalize()} agent. Please help with the current task."}
        
        # Create a system message as user message with role prefix for LM Studio compatibility
        role_prefix = f"You are acting as the {prompt_name.capitalize()} agent. "
        system_as_user = {"role": "user", "content": role_prefix + system_prompt}
        
        # Return the single message
        return system_as_user
        
    except Exception as e:
        logger.exception(f"Error applying template {prompt_name}")
        # Return a minimal working message
        return {"role": "user", "content": f"You are the {prompt_name} agent. Please help with the current request."}

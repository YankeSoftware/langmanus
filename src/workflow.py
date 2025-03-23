import logging
import traceback
from src.config import TEAM_MEMBERS
import os
from typing import Dict, Union, List, Any
from src.utils.message_utils import normalize_message, normalize_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default level is INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def enable_debug_logging():
    """Enable debug level logging for more detailed execution information."""
    logging.getLogger("src").setLevel(logging.DEBUG)
    logging.getLogger("langchain").setLevel(logging.DEBUG)
    logging.getLogger("litellm").setLevel(logging.DEBUG)


logger = logging.getLogger(__name__)

# Do NOT create the graph here - it creates circular imports
# Instead, create it lazily in each function that needs it


def run_agent_workflow(user_input: str, debug: bool = False):
    """Run the agent workflow with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging

    Returns:
        The final state after the workflow completes
    """
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty")

    if debug:
        enable_debug_logging()
        logger.debug("Debug logging enabled")

    logger.info(f"Starting workflow with user input: {user_input}")
    
    try:
        # Import graph builder here to avoid circular imports
        from src.graph import build_graph
        graph = build_graph()
        
        # Prepare the initial state
        initial_state = {
            # Constants
            "TEAM_MEMBERS": TEAM_MEMBERS,
            # Runtime Variables
            "messages": [{"role": "user", "content": user_input}],
            "deep_thinking_mode": True,
            "search_before_planning": True,
            # Initialize empty state fields
            "actions": "",
            "plan": "",
            "tasks": ""
        }
        
        logger.debug(f"Initial state: {initial_state}")
        
        # Run the workflow with the user input
        result = graph.invoke(initial_state)
        logger.debug(f"Final workflow state: {result}")
        logger.info("Workflow completed successfully")
        return result
        
    except Exception as e:
        logger.exception("Workflow execution failed")
        error_msg = str(e)
        traceback_text = traceback.format_exc()
        logger.debug(f"Error traceback: {traceback_text}")
        
        # Handle common LM Studio errors with more helpful messages
        if "Only user and assistant roles are supported" in error_msg:
            logger.error("LM Studio compatibility error: Only user and assistant roles are supported")
            raise RuntimeError(
                "LM Studio requires 'user' and 'assistant' roles only. "
                "This may be a message format issue. Check that all messages sent to the model "
                "use only these two roles."
            )
        elif "response_format.type" in error_msg:
            logger.error("LM Studio JSON format error")
            raise RuntimeError(
                f"LM Studio has different JSON formatting requirements. "
                f"This should be fixed with our custom LiteLLM adapter. "
                f"If you see this error, please report it."
            )
        elif "OpenAI API" in error_msg or "status code: 400" in error_msg or "status code: 404" in error_msg:
            logger.error("LM Studio API communication error")
            raise RuntimeError(
                f"Error communicating with LM Studio API. Please ensure:\n"
                f"1. LM Studio is running with Mistral-7B-Instruct-v0.3 or compatible model\n"
                f"2. The API is accessible at http://localhost:1234/v1\n"
                f"3. In LM Studio settings, OpenAI API option is enabled\n"
                f"Error details: {error_msg}"
            )
        else:
            # Create a more informative error message for other errors
            raise RuntimeError(
                f"Workflow failed with error: {error_msg}\n"
                f"Check the logs for more details."
            ) from e


def diagnose_environment(verbose: bool = False) -> Dict[str, bool]:
    """
    Run a diagnostic check on the environment to ensure all components are working.
    
    Args:
        verbose: Whether to print diagnostic information
        
    Returns:
        Dict of component names and their status (True if working, False otherwise)
    """
    results = {}
    
    # Check LM Studio connection
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        results["lm_studio"] = response.status_code == 200
        if verbose:
            if results["lm_studio"]:
                logger.info("✅ LM Studio: Connected successfully")
                # Get available models
                models = response.json().get("data", [])
                model_names = [model.get("id") for model in models]
                logger.info(f"   Available models: {', '.join(model_names)}")
            else:
                logger.error(f"❌ LM Studio: Connection failed (status code: {response.status_code})")
    except Exception as e:
        results["lm_studio"] = False
        if verbose:
            logger.error(f"❌ LM Studio: Connection error - {str(e)}")
    
    # Check for required environment variables
    env_vars = [
        "BRAVE_API_KEY",
        "OPENAI_API_KEY"
    ]
    
    results["env_variables"] = {}
    missing_vars = []
    
    for var in env_vars:
        has_var = var in os.environ and os.environ[var]
        results["env_variables"][var] = has_var
        if not has_var:
            missing_vars.append(var)
    
    if verbose:
        if not missing_vars:
            logger.info("✅ Environment Variables: All required variables are set")
        else:
            logger.warning(f"⚠️ Environment Variables: Missing: {', '.join(missing_vars)}")
            logger.info("   Some features may be limited without these variables")
    
    # Check memory system
    try:
        from src.integration import memory
        results["memory_system"] = memory.is_available()
        if verbose:
            if results["memory_system"]:
                logger.info("✅ Memory System: Available")
            else:
                logger.warning("⚠️ Memory System: Not available")
    except Exception as e:
        results["memory_system"] = False
        if verbose:
            logger.error(f"❌ Memory System: Error - {str(e)}")
            
    # Check RLHF capability
    try:
        from src.integration import feedback
        results["rlhf_system"] = feedback.is_available()
        if verbose:
            if results["rlhf_system"]:
                logger.info("✅ RLHF System: Available")
            else:
                logger.warning("⚠️ RLHF System: Not available")
    except Exception as e:
        results["rlhf_system"] = False
        if verbose:
            logger.error(f"❌ RLHF System: Error - {str(e)}")
    
    return results


if __name__ == "__main__":
    # Import graph builder here to avoid circular imports
    from src.graph import build_graph
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())

import logging
import traceback
import os
import sys
import requests
from src.config import TEAM_MEMBERS
from typing import Dict, Union, List, Any, Optional, Tuple, cast
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


def run_agent_workflow(user_input: str, debug: bool = False) -> Dict:
    """Run the agent workflow with the given user input."""
    # Set up debug logging if requested
    if debug:
        enable_debug_logging()
        
    logger.info(f"Starting workflow with user input: {user_input}")
    
    # Initialize the agent graph and get search tools
    search_tool = get_search_tool()
    logger.debug(f"Search tool: {search_tool}")
    if search_tool is None:
        logger.error("No search tool available - check Brave or Tavily API keys")
        raise RuntimeError(
            "No search tool is available. Please check that either BRAVE_API_KEY "
            "or TAVILY_API_KEY environment variables are set correctly."
        )
    
    # Create initial state
    state = initial_state(user_input)
    
    # Create agent graph
    builder = workflow_builder()
    graph = builder.compile()
    
    # Track the token count to avoid context overflows
    max_tokens = 3500  # Safe limit for Mistral 7B model (4096 max)
    
    try:
        # Run the workflow
        result = graph.invoke(state)
        
        # Check final message count and prune if needed
        if "messages" in result and len(result["messages"]) > 15:
            logger.warning(f"Message count ({len(result['messages'])}) exceeds safe limit, pruning...")
            
            # Keep first (user query) and last N messages for coherence
            pruned_messages = [result["messages"][0]]  # Keep the user query
            pruned_messages.extend(result["messages"][-10:])  # Keep the last 10 messages
            
            # Update result with pruned messages
            result["messages"] = pruned_messages
            logger.info(f"Pruned message count to {len(result['messages'])}")
        
        return result
    except Exception as e:
        logger.exception(f"Error running workflow: {e}")
        raise


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


def initial_state(user_input: str) -> Dict:
    """Create the initial state for the workflow."""
    if not user_input or not user_input.strip():
        raise ValueError("User input cannot be empty")
        
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
    return initial_state


def workflow_builder():
    """Create and return the workflow builder."""
    # Import graph builder here to avoid circular imports
    from src.graph import build_graph
    return build_graph()


def get_search_tool():
    """Get the appropriate search tool based on available integrations."""
    from src.tools.search import default_search_tool
    return default_search_tool


if __name__ == "__main__":
    # Import graph builder here to avoid circular imports
    from src.graph import build_graph
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())

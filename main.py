"""
Entry point script for the LangManus system.
"""

import logging
import os
import sys
import time
import traceback
import requests
from typing import List, Dict, Any, Optional
from src.workflow import (
    run_agent_workflow, 
    enable_debug_logging, 
    diagnose_environment
)
from src.utils.message_utils import normalize_message, normalize_messages
from src.integration import memory
from src.integration import feedback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_lm_studio_running() -> tuple[bool, List[str]]:
    """
    Check if LM Studio appears to be running and accessible.
    
    Returns:
        Tuple containing:
        - Boolean indicating if LM Studio is running and accessible
        - List of available model names
    """
    # Get timeout from environment variable or use default
    timeout = float(os.environ.get("LM_STUDIO_TIMEOUT", "2.0"))
    
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=timeout)
        if response.status_code == 200:
            logger.info("Successfully connected to LM Studio API")
            # Get available models
            models = response.json().get("data", [])
            model_names = [model.get("id") for model in models]
            logger.info(f"Available LM Studio models: {', '.join(model_names)}")
            return True, model_names
    except requests.exceptions.ConnectionError:
        logger.warning("Connection refused - LM Studio is not running or not accessible")
    except requests.exceptions.Timeout:
        logger.warning(f"Connection to LM Studio timed out after {timeout} seconds")
    except Exception as e:
        logger.warning(f"Could not connect to LM Studio: {type(e).__name__}: {e}")
    return False, []

def check_openai_credentials() -> bool:
    """
    Check if OpenAI API credentials are configured.
    
    Returns:
        Boolean indicating if OpenAI credentials are properly configured
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OpenAI API key not found in environment variables")
        return False
    
    # Optional basic validation of key format
    if not api_key.startswith("sk-") or len(api_key) < 20:
        logger.warning("OpenAI API key has invalid format")
        return False
    
    logger.info("OpenAI credentials found")
    return True

def get_user_feedback(user_query, response):
    """
    Ask for and record user feedback if the feedback system is available.
    
    Args:
        user_query: The original user query
        response: The system's response to the query
    """
    if not feedback.is_available():
        return
        
    try:
        print("\n--- Feedback ---")
        print("How would you rate the response? (1-5, with 5 being best)")
        print("Enter 0 to skip feedback")
        
        # Status indicator for HITL
        print("[WAITING FOR HUMAN INPUT]")
        
        while True:
            try:
                score_input = input("Score (0-5): ").strip()
                score = int(score_input)
                if 0 <= score <= 5:
                    break
                print("Please enter a number between 0 and 5.")
            except ValueError:
                print("Please enter a valid number.")
                
        if score == 0:
            print("Feedback skipped.")
            return
            
        # Optionally get text feedback
        feedback_text = input("Any additional comments? (optional): ").strip()
        
        # Record the feedback
        success = feedback.record_feedback(
            user_input=user_query,
            model_response=response,
            score=score,
            feedback_text=feedback_text
        )
        
        if success:
            print("Thank you for your feedback!")
        else:
            logger.error("Failed to record feedback")
            print("Failed to record feedback.")
            
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        print("Failed to collect feedback due to an error.")

def setup_api_environment() -> Dict[str, Any]:
    """
    Set up the API environment based on available services and credentials.
    
    This function:
    1. Checks for LM Studio availability (prioritized)
    2. Falls back to OpenAI if LM Studio is not available
    3. Returns a properly formatted API configuration
    4. Sets environment variables for API access
    
    Returns:
        API configuration dictionary
    """
    api_config = {}
    
    # First check for LM Studio
    lm_studio_running, available_models = check_lm_studio_running()
    
    if lm_studio_running:
        logger.info("LM Studio detected - configuring for local model usage")
        
        # Check if Mistral 7B is available (preferred model)
        mistral_model = None
        for model in available_models:
            if "mistral-7b-instruct" in model.lower():
                mistral_model = model
                logger.info(f"Found preferred Mistral model: {mistral_model}")
                break
        
        # Set up configuration for LM Studio
        api_config = {
            "model": mistral_model or available_models[0],
            "api_base": "http://localhost:1234/v1",
            "api_type": "lm_studio",
            "temperature": 0.7,
            "max_tokens": 1500,
            "top_p": 0.95
        }
        
        # Set environment variables for LiteLLM and LangChain
        os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
        os.environ["OPENAI_API_KEY"] = "lm-studio"  # Dummy key for LM Studio
        os.environ["DEFAULT_MODEL"] = api_config["model"]
        
        # Set special environment variable to indicate LM Studio is being used
        os.environ["USE_LM_STUDIO"] = "true"
        
        logger.info(f"API environment configured for LM Studio with model: {api_config['model']}")
        return api_config
    
    # Then check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key and len(openai_api_key) > 20:  # Basic validation
        logger.info("OpenAI API key detected - configuring for OpenAI usage")
        
        # Set up configuration for OpenAI
        api_config = {
            "model": os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo"),
            "api_type": "openai",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.95
        }
        
        # Set environment variables to ensure consistency
        os.environ["DEFAULT_MODEL"] = api_config["model"]
        os.environ["USE_LM_STUDIO"] = "false"
        
        logger.info(f"API environment configured for OpenAI with model: {api_config['model']}")
        return api_config
    
    # If neither is available, log a warning and return a minimal configuration
    logger.warning("No suitable API provider (LM Studio or OpenAI) detected!")
    logger.warning("The system may fail when trying to use language models.")
    logger.warning("Please start LM Studio or set an OpenAI API key.")
    
    # Provide a minimal configuration that will likely fail but has the correct structure
    api_config = {
        "model": "none",
        "api_type": "none",
        "temperature": 0.7,
        "max_tokens": 1000,
    }
    
    return api_config

if __name__ == "__main__":
    # Always enable debug logging for better diagnostics
    enable_debug_logging()
    
    # Parse command line arguments
    enable_hitl = True  # Default to enabled
    
    # Remove debug flag if present
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
    
    # HITL flags
    if "--no-hitl" in sys.argv:
        enable_hitl = False
        sys.argv.remove("--no-hitl")
    
    if "--hitl" in sys.argv:
        enable_hitl = True
        sys.argv.remove("--hitl")
    
    # Run diagnostics if requested
    if "--diagnostics" in sys.argv or "--diag" in sys.argv:
        if "--diagnostics" in sys.argv:
            sys.argv.remove("--diagnostics")
        if "--diag" in sys.argv:
            sys.argv.remove("--diag")
        
        print("\n=== 🦜🤖 LangManus Diagnostics ===")
        print("Running system checks...\n")
        
        # Run environment diagnostics
        results = diagnose_environment(verbose=True)
        
        # Print summary
        print("\n=== Diagnostics Summary ===")
        if all(results.values()):
            print("✅ All systems operational")
        else:
            print("⚠️ Some components have issues:")
            for component, status in results.items():
                if isinstance(status, dict):
                    for item, item_status in status.items():
                        print(f"  {'✅' if item_status else '❌'} {component}: {item}")
                else:
                    print(f"  {'✅' if status else '❌'} {component}")
        
        if len(sys.argv) <= 1:
            sys.exit(0)
    
    # Setup the API environment
    api_config = setup_api_environment()

    # Check if memory system is available
    memory_available = memory.is_available()
    if memory_available:
        logger.info("Memory system is available and integrated.")
    
    # Check if we have arguments or need to prompt for input
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        print("\n=== 🦜🤖 LangManus AI System ===")
        print("Your AI agent team powered by LLM backend")
        print("---------------------------------------------------")
        
        # Display available API providers
        print(f"\nAPI Configuration:")
        
        if api_config["api_type"] == "lm_studio":
            print(f"✅ LM Studio: Available at {api_config['api_base']}")
            if api_config["model"]:
                print(f"   Models: {api_config['model']}")
        else:
            print("❌ LM Studio: Not available")
            
        if api_config["api_type"] == "openai":
            print(f"✅ OpenAI API: Configured")
        else:
            print("❌ OpenAI API: Not configured")
            
        print(f"\nUsing: {api_config['api_type'].upper()} as primary provider")
        
        # Display warnings for missing providers
        if api_config["api_type"] == "lm_studio" and not api_config["model"]:
            print("\n⚠️  WARNING: Primary provider (LM Studio) is not available!")
            print("  Please start LM Studio with a compatible model")
            print("  Run with --diagnostics for more detailed system checks.")
        
        # Display HITL status
        if enable_hitl:
            print("\n✅ Human-in-the-Loop mode is ENABLED")
            print("   You'll be asked to approve plans before execution")
            print("   Use --no-hitl flag to disable this feature")
        else:
            print("\n⚠️ Human-in-the-Loop mode is DISABLED")
            print("   Use --hitl flag to enable this feature")

        user_query = input("\nEnter your query: ")

    # Proceed to workflow execution
    try:
        print("\n=== Processing your query... ===")
        
        # Set execution start time for timeout detection
        start_time = time.time()
        max_execution_time = float(os.environ.get("MAX_EXECUTION_TIME", "300"))  # 5 minutes default
        
        # Run the agent workflow with configured settings
        result = run_agent_workflow(
            user_query, 
            debug=True, 
            enable_hitl=enable_hitl,
            api_config=api_config
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"Query processed in {execution_time:.2f} seconds")
        
        print("\n=== Result ===")
        
        # Format the conversation history in a readable way 
        if "messages" in result and result["messages"]:
            for message in result["messages"]:
                # Convert to standard format
                normalized = normalize_message(message)
                
                if normalized["role"] == "user":
                    print(f"\n🧑 User: {normalized['content']}")
                else:
                    print(f"\n🤖 Assistant: {normalized['content']}")
        
        # Get feedback from the user if workflow was successful
        if result.get("messages") and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            normalized = normalize_message(last_message)
            get_user_feedback(user_query, normalized["content"])
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
        print(f"\n❌ Connection Error: Could not connect to the LLM provider")
        print("\nPossible solutions:")
        print("1. Make sure LM Studio is running (if using local models)")
        print("2. Check your internet connection (if using OpenAI)")
        print("3. Run with --diagnostics to check system configuration")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        if "Error applying template" in str(e):
            print(f"\n❌ Template Error: {e}")
            print("\nThis is likely due to a missing or invalid template variable.")
            print("Check the logs for more details and consider running with --diagnostics.")
        else:
            print(f"\n❌ Error: {e}")
    except RuntimeError as e:
        logger.error(f"Workflow execution failed: {e}")
        print(f"\n❌ Error: {e}")
        
        # Check for specific error types and give helpful messages
        error_str = str(e).lower()
        if "json" in error_str and "format" in error_str:
            print("\nThis appears to be a JSON format issue with the LLM provider.")
            print("Please ensure you're using a model that supports the required format.")
        elif ("lm studio" in error_str or "openai" in error_str) and ("connection" in error_str or "api" in error_str):
            print("\nThis appears to be an issue connecting to the LLM provider.")
            print("Please check that the service is running and your credentials are correct.")
    except KeyboardInterrupt:
        print("\n\n⚠️ Process interrupted by user")
        print("Stopping workflow execution...")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print(f"Type: {type(e).__name__}")
        print(f"Error details: {traceback.format_exc()}")
        print("Check the logs for more details or run with --debug for verbose logging.")

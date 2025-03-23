"""
Entry point script for the LangManus system.
"""

import logging
import os
import sys
import traceback
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

def check_lm_studio_running():
    """Check if LM Studio appears to be running and accessible."""
    import requests
    try:
        response = requests.get("http://localhost:1234/v1/models", timeout=2)
        if response.status_code == 200:
            logger.info("Successfully connected to LM Studio API")
            return True
    except Exception as e:
        logger.warning(f"Could not connect to LM Studio: {e}")
    return False

def get_user_feedback(user_query, response):
    """Ask for and record user feedback if the feedback system is available."""
    if not feedback.is_available():
        return
        
    try:
        print("\n--- Feedback ---")
        print("How would you rate the response? (1-5, with 5 being best)")
        print("Enter 0 to skip feedback")
        
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
            print("Failed to record feedback.")
            
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        print("Failed to collect feedback due to an error.")

if __name__ == "__main__":
    # Always enable debug logging for better diagnostics
    enable_debug_logging()
    
    # Remove debug flag if present
    if "--debug" in sys.argv:
        sys.argv.remove("--debug")
    
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
        
    # Check if LM Studio appears to be running
    lm_studio_running = check_lm_studio_running()

    # Check if memory system is available
    if memory.is_available():
        logger.info("Memory system is available and integrated.")
    
    # Check if we have arguments or need to prompt for input
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        print("\n=== 🦜🤖 LangManus AI System ===")
        print("Your AI agent team powered by LM Studio")
        print("---------------------------------------------------")
        # Notify about LM Studio requirement
        print("NOTE: Make sure LM Studio is running with Mistral-7B-Instruct-v0.3 loaded")
        print("      Server should be running on http://localhost:1234")

        if not lm_studio_running:
            print("\n⚠️  WARNING: LM Studio connection not detected! Please ensure:")
            print("  1. LM Studio is running with the server started")
            print("  2. Mistral-7B-Instruct-v0.3 model is loaded")
            print("  3. OpenAI API mode is enabled in settings")
            print("  4. Port 1234 is being used")
            print("\nProceeding anyway, but errors may occur.")
            print("Run with --diagnostics for more detailed system checks.")

        user_query = input("\nEnter your query: ")

    # Just proceed to workflow execution if either memory and LM Studio are ready
    try:
        print("\n=== Processing your query... ===")
        result = run_agent_workflow(user_query, debug=True)
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
        
        # Get feedback from the user
        if result.get("messages") and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            normalized = normalize_message(last_message)
            get_user_feedback(user_query, normalized["content"])
            
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
            print("\nThis appears to be a JSON format issue with LM Studio.")
            print("Please ensure you're using a Mistral model that supports the required format.")
        elif "lm studio" in error_str and ("connection" in error_str or "api" in error_str):
            print("\nThis appears to be an issue connecting to LM Studio.")
            print("Please check that LM Studio is running with the server started.")
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print(f"Type: {type(e).__name__}")
        print(f"Error details: {traceback.format_exc()}")
        print("Check the logs for more details or run with --debug for verbose logging.")

#!/usr/bin/env python3
"""
Basic example of using LangManus with LM Studio
"""

import os
import sys
import logging

# Set environment variable to enable LM Studio compatibility mode
os.environ["USE_LM_STUDIO"] = "true"

# Add the repository root to the Python path to allow importing the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workflow import run_agent_workflow, enable_debug_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run a simple LangManus workflow with LM Studio."""
    
    # Enable debug logging for detailed output
    enable_debug_logging()
    
    # Get user input
    print("LangManus with LM Studio - Basic Example")
    print("=======================================")
    print("Using this example, you can interact with LangManus using your local LM Studio model.")
    print("Make sure LM Studio is running with the Mistral-7B-Instruct-v0.3 model.")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get user input
        user_input = input("\nEnter your query: ")
        
        # Exit condition
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting...")
            break
        
        # Skip empty input
        if not user_input.strip():
            continue
        
        try:
            # Run the workflow
            print("\nProcessing your query...\n")
            result = run_agent_workflow(user_input, debug=True)
            
            # Display the final result
            print("\n--- RESULT ---")
            
            # Extract all assistant messages from the result
            for message in result.get("messages", []):
                if isinstance(message, dict) and message.get("role") == "assistant":
                    print(f"{message.get('content', '')}\n")
            
            print("--- END OF RESULT ---\n")
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            print(f"\nError: {e}")


if __name__ == "__main__":
    main() 
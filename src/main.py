"""
Entry point script for the LangManus system.
"""

import logging
import os
import sys
import traceback
from src.workflow import run_agent_workflow, enable_debug_logging, diagnose_environment
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
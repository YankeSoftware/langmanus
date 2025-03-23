"""
Feedback integration module providing RLHF (Reinforcement Learning from Human Feedback) capabilities.

This module allows for collecting user feedback on model interactions and storing
it for potential future training or adaptation.
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)

# Path to store feedback data
FEEDBACK_DIR = Path("./data/feedback")

def is_available() -> bool:
    """
    Check if the feedback system is available.
    
    Returns:
        bool: True if the feedback system can be used, False otherwise
    """
    try:
        # Ensure feedback directory exists
        os.makedirs(FEEDBACK_DIR, exist_ok=True)
        return True
    except Exception as e:
        logger.warning(f"Feedback system not available: {e}")
        return False

def record_feedback(
    user_input: str,
    model_response: str,
    score: int,
    feedback_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Record user feedback on a model interaction.
    
    Args:
        user_input: The original user query
        model_response: The model's response to the query
        score: Feedback score (1-5, with 5 being most positive)
        feedback_text: Optional textual feedback
        metadata: Optional additional metadata about the interaction
        
    Returns:
        bool: True if feedback was successfully recorded, False otherwise
    """
    if not is_available():
        logger.warning("Cannot record feedback: feedback system not available")
        return False
        
    try:
        # Validate score
        if not isinstance(score, int) or score < 1 or score > 5:
            logger.error(f"Invalid feedback score: {score}. Must be integer from 1-5.")
            return False
            
        # Create feedback record
        timestamp = datetime.now().isoformat()
        record_id = f"{timestamp.replace(':', '-').replace('.', '-')}"
        
        feedback_record = {
            "id": record_id,
            "timestamp": timestamp,
            "user_input": user_input,
            "model_response": model_response,
            "score": score,
            "feedback_text": feedback_text or "",
            "metadata": metadata or {}
        }
        
        # Write to file
        filename = f"{record_id}.json"
        filepath = FEEDBACK_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(feedback_record, f, indent=2)
            
        logger.info(f"Recorded feedback with ID {record_id} and score {score}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        return False

def get_recent_feedback(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recent feedback records.
    
    Args:
        limit: Maximum number of records to retrieve
        
    Returns:
        List of feedback records sorted by recency
    """
    if not is_available():
        logger.warning("Cannot retrieve feedback: feedback system not available")
        return []
        
    try:
        # Get all feedback files
        feedback_files = list(FEEDBACK_DIR.glob("*.json"))
        
        # Sort by creation time (most recent first)
        feedback_files.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # Limit number of files
        feedback_files = feedback_files[:limit]
        
        # Read and parse files
        feedback_records = []
        for file in feedback_files:
            try:
                with open(file, 'r') as f:
                    record = json.load(f)
                    feedback_records.append(record)
            except Exception as e:
                logger.error(f"Error reading feedback file {file}: {e}")
                
        return feedback_records
        
    except Exception as e:
        logger.error(f"Failed to retrieve feedback: {e}")
        return []

def get_average_score() -> Optional[float]:
    """
    Calculate the average feedback score across all records.
    
    Returns:
        Average score or None if no feedback is available
    """
    if not is_available():
        return None
        
    try:
        # Get all feedback files
        feedback_files = list(FEEDBACK_DIR.glob("*.json"))
        
        if not feedback_files:
            logger.info("No feedback records available")
            return None
            
        # Calculate average score
        total_score = 0
        count = 0
        
        for file in feedback_files:
            try:
                with open(file, 'r') as f:
                    record = json.load(f)
                    total_score += record.get("score", 0)
                    count += 1
            except Exception as e:
                logger.error(f"Error reading feedback file {file}: {e}")
                
        if count == 0:
            return None
            
        return total_score / count
        
    except Exception as e:
        logger.error(f"Failed to calculate average score: {e}")
        return None 
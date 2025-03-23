"""
Server script for running the LangManus API.
"""

import logging
import uvicorn
import sys
import os
from src.integration import memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Check if memory system is available
    if memory.is_available():
        logger.info("Memory system is available and integrated.")
    
    # Check if LM Studio is likely running (we can't verify connection yet)
    print("\n=== 🦜🤖 LangManus API Server ===")
    print("NOTE: Make sure LM Studio is running with Mistral-7B-Instruct-v0.3 loaded")
    print("      Server should be running on http://localhost:1234")
    print("-----------------------------------------------------------")
        
    logger.info("Starting LangManus API server")
    reload = True
    if sys.platform.startswith("win"):
        reload = False
    
    try:
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=reload,
            log_level="info",
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        print(f"\nAn error occurred starting the server: {str(e)}")
        print("Check that port 8000 is available and that you have the necessary permissions.")

import os
import logging
import sys
from datetime import datetime

# Configure logging
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.environ.get("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_FILE = os.environ.get("LOG_FILE", "")

# Ensure log directory exists
if LOG_FILE and not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
    except Exception as e:
        print(f"Error creating log directory: {e}")

# Configure the root logger
logger = logging.getLogger("langmanus")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(getattr(logging, LOG_LEVEL))
console_formatter = logging.Formatter(LOG_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Add file handler if LOG_FILE is specified
if LOG_FILE:
    try:
        file_path = os.path.join(LOG_DIR, LOG_FILE)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, LOG_LEVEL))
        file_formatter = logging.Formatter(LOG_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging: {e}")
        logger.error(f"Error setting up file logging: {e}")

# Create a performance logger for metrics
perf_logger = logging.getLogger("langmanus.performance")
perf_logger.setLevel(logging.INFO)

# Add a separate file handler for performance metrics if LOG_DIR exists
if os.path.exists(LOG_DIR):
    try:
        perf_file = os.path.join(LOG_DIR, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        perf_handler = logging.FileHandler(perf_file)
        perf_formatter = logging.Formatter("%(asctime)s - %(message)s")
        perf_handler.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_handler)
        
        # Also add performance logs to console with DEBUG level
        perf_console = logging.StreamHandler(sys.stdout)
        perf_console.setLevel(logging.DEBUG)
        perf_console.setFormatter(perf_formatter)
        perf_logger.addHandler(perf_console)
    except Exception as e:
        logger.error(f"Error setting up performance logging: {e}")

def log_performance(agent, duration, cache_hit=False, message_tokens=0):
    """Log performance metrics for an agent execution"""
    try:
        perf_logger.info(f"{agent},{duration:.4f},{cache_hit},{message_tokens}")
    except Exception as e:
        logger.error(f"Error logging performance: {e}")

# Set logging levels for noisy external libraries
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Export the main logger
__all__ = ['logger', 'log_performance'] 
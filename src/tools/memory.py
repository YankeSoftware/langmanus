import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from langchain.tools.base import BaseTool

from src.integration import memory
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

MEMORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "memory")

def is_available() -> bool:
    """Check if memory system is available and accessible."""
    try:
        if not os.path.exists(MEMORY_DIR):
            os.makedirs(MEMORY_DIR, exist_ok=True)
            logger.info(f"Created memory directory: {MEMORY_DIR}")
        
        # Quick write/read test to verify filesystem access
        test_file = os.path.join(MEMORY_DIR, '.test_access')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        
        return True
    except Exception as e:
        logger.warning(f"Memory system unavailable: {str(e)}")
        return False

def save_to_memory(key: str, data: Any) -> bool:
    """
    Save data to memory with error handling.
    
    Args:
        key: Unique identifier for the memory
        data: Data to store (will be JSON serialized)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        if not os.path.exists(MEMORY_DIR):
            os.makedirs(MEMORY_DIR, exist_ok=True)
        
        # Create filename from key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        filename = f"{safe_key}.json"
        filepath = os.path.join(MEMORY_DIR, filename)
        
        # Add timestamp for versioning
        if isinstance(data, dict):
            data = data.copy()  # Make a copy to avoid modifying the original
            data["_timestamp"] = time.time()
            data["_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Write data to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.debug(f"Saved memory: {key}")
        return True
    except Exception as e:
        logger.error(f"Error saving to memory: {str(e)}")
        return False

def load_from_memory(key: str) -> Optional[Any]:
    """
    Load data from memory with error handling.
    
    Args:
        key: Unique identifier for the memory
        
    Returns:
        Loaded data or None if not found or error
    """
    try:
        # Create filename from key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        filename = f"{safe_key}.json"
        filepath = os.path.join(MEMORY_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.debug(f"Memory not found: {key}")
            return None
            
        # Read data from file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        logger.debug(f"Loaded memory: {key}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding memory file for {key}: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading from memory: {str(e)}")
        return None

def list_memories() -> List[str]:
    """
    List all available memories with error handling.
    
    Returns:
        List of memory keys
    """
    try:
        if not os.path.exists(MEMORY_DIR):
            return []
            
        # Get all JSON files in the memory directory
        memory_files = [f for f in os.listdir(MEMORY_DIR) if f.endswith('.json')]
        
        # Convert filenames back to keys
        keys = [os.path.splitext(f)[0] for f in memory_files]
        
        # Remove any temporary files
        keys = [k for k in keys if not k.startswith('.')]
        
        return keys
    except Exception as e:
        logger.error(f"Error listing memories: {str(e)}")
        return []

def clear_memory(key: str) -> bool:
    """
    Delete a specific memory with error handling.
    
    Args:
        key: Unique identifier for the memory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create filename from key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        filename = f"{safe_key}.json"
        filepath = os.path.join(MEMORY_DIR, filename)
        
        if not os.path.exists(filepath):
            logger.debug(f"Memory not found: {key}")
            return False
            
        # Delete file
        os.remove(filepath)
        logger.debug(f"Cleared memory: {key}")
        return True
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        return False

class MemoryStoreTool(BaseTool):
    """Tool for storing information in the memory system."""
    
    name: str = "memory_store"
    description: str = "Store important information in the memory system for future reference."
    
    def _run(self, content: str, source: str = "langmanus", tags: Optional[List[str]] = None) -> str:
        """Store information in the memory system."""
        logger.info(f"Storing memory: {content[:50]}...")
        
        if not memory.is_available():
            return "Memory system is not available. Information will not be stored, but has been acknowledged."
            
        try:
            memory_id = memory.store_memory(content, source, tags or ["important"])
            
            if memory_id:
                return f"Memory stored successfully with ID: {memory_id}"
            else:
                return "Failed to store memory. Information was received but not saved to long-term memory."
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"Error storing memory: {str(e)}. Information was received but not saved to long-term memory."
    
    async def _arun(self, content: str, source: str = "langmanus", tags: Optional[List[str]] = None) -> str:
        """Store information in the memory system asynchronously."""
        return self._run(content, source, tags)


class MemoryRetrieveTool(BaseTool):
    """Tool for retrieving information from the memory system."""
    
    name: str = "memory_retrieve"
    description: str = "Retrieve relevant information from the memory system based on a query."
    
    def _run(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve information from the memory system."""
        logger.info(f"Retrieving memories for query: {query}")
        
        if not memory.is_available():
            return [{"message": "Memory system is not available. Proceeding without memory context."}]
            
        try:
            memories = memory.retrieve_memories(query, limit)
            
            if not memories:
                return [{"message": "No relevant memories found. This appears to be new information."}]
                
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return [{"message": f"Error retrieving memories: {str(e)}. Proceeding without memory context."}]
    
    async def _arun(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve information from the memory system asynchronously."""
        return self._run(query, limit)

# Create memory tools
memory_store_tool = None
memory_retrieve_tool = None

try:
    # Initialize the memory tools if memory system is available
    if memory.is_available():
        LoggedMemoryStore = create_logged_tool(MemoryStoreTool)
        LoggedMemoryRetrieve = create_logged_tool(MemoryRetrieveTool)
        
        memory_store_tool = LoggedMemoryStore(name="memory_store")
        memory_retrieve_tool = LoggedMemoryRetrieve(name="memory_retrieve")
        
        logger.info("Memory tools initialized successfully")
    else:
        logger.warning("Memory system is not available. Memory tools will not be initialized.")
except Exception as e:
    logger.error(f"Failed to initialize memory tools: {e}")
    # Ensure these are None if initialization fails
    memory_store_tool = None 
    memory_retrieve_tool = None 
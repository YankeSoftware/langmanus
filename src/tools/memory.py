import logging
from typing import Dict, List, Any, Optional
from langchain.tools.base import BaseTool

from src.integration import memory
from .decorators import create_logged_tool

logger = logging.getLogger(__name__)

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
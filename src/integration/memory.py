import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class SimpleMemory:
    """Simple in-memory storage system for LangManus."""
    
    def __init__(self):
        """Initialize the simple memory system."""
        # In-memory storage for memories
        self.memories = []
        # Always available (no external dependencies)
        self.is_available_flag = True
        logger.info("Simple memory system initialized")
    
    def is_available(self) -> bool:
        """Check if memory system is available."""
        return self.is_available_flag
    
    def store_memory(self, content: str, source: str = "langmanus", tags: List[str] = None) -> str:
        """Store a new memory."""
        memory_id = f"memory_{len(self.memories) + 1}"
        
        # Create a new memory entry
        memory = {
            "id": memory_id,
            "content": content,
            "source": source,
            "tags": tags or ["langmanus"],
            "created_at": "2023-01-01T00:00:00Z"  # Placeholder timestamp
        }
        
        # Add to memory store
        self.memories.append(memory)
        logger.info(f"Stored memory with ID: {memory_id}")
        
        return memory_id
    
    def retrieve_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on a simple keyword match."""
        if not self.memories:
            logger.info("No memories to retrieve")
            return []
            
        # Simple keyword matching (not semantic search)
        query_terms = query.lower().split()
        
        # Score memories based on keyword matches
        scored_memories = []
        for memory in self.memories:
            score = 0
            content_lower = memory["content"].lower()
            
            # Count how many query terms appear in the content
            for term in query_terms:
                if term in content_lower:
                    score += 1
            
            # Only include if there's at least one match
            if score > 0:
                relevance = score / len(query_terms)
                scored_memories.append({
                    "id": memory["id"],
                    "title": f"Memory {memory['id']}",
                    "content": memory["content"],
                    "relevance": relevance,
                    "tags": memory["tags"]
                })
        
        # Sort by relevance and limit results
        result = sorted(scored_memories, key=lambda x: x["relevance"], reverse=True)[:limit]
        logger.info(f"Retrieved {len(result)} memories for query: {query}")
        
        return result

# Initialize the global instance
memory = SimpleMemory()

# For backward compatibility
def is_available():
    """Check if memory system is available."""
    return memory.is_available()

# Test function
if __name__ == "__main__":
    print("Simple memory system test:")
    
    # Store some test memories
    memory_id1 = memory.store_memory(
        "Python is a high-level programming language known for its readability.",
        tags=["python", "programming"]
    )
    memory_id2 = memory.store_memory(
        "JavaScript is a programming language commonly used for web development.",
        tags=["javascript", "web", "programming"]
    )
    
    # Test retrieval
    memories = memory.retrieve_memories("python programming")
    print(f"Retrieved {len(memories)} memories:")
    for mem in memories:
        print(f"- [{mem['relevance']:.2f}] {mem['content']}") 
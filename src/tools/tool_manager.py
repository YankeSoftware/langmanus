import logging
from typing import List, Dict, Any, Optional
from langchain.tools import Tool, BaseTool
from src.tools.search import brave_search
from src.tools.crawler import web_crawler

logger = logging.getLogger(__name__)

class ToolManager:
    """
    Manages all tools available to agents in the system.
    Provides a centralized way to access and configure tools.
    """
    
    def __init__(self):
        """Initialize the tool manager with all available tools."""
        self.available_tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools with the manager."""
        # Register search tools
        self._register_tool(
            Tool(
                name="brave_search",
                description="Search the web for information using Brave Search.",
                func=brave_search.run,
            )
        )
        
        # Register web crawler tools
        self._register_tool(
            Tool(
                name="get_webpage_content", 
                description="Get the content of a webpage by URL.",
                func=lambda url: web_crawler.run("get_content", url),
            )
        )
        
        self._register_tool(
            Tool(
                name="follow_webpage_links",
                description="Follow links from a starting URL to explore related content.",
                func=lambda url, depth=1, max_pages=3: web_crawler.run("follow_links", url, depth=depth, max_pages=max_pages),
            )
        )
        
        self._register_tool(
            Tool(
                name="search_webpage_content",
                description="Search for specific terms within a webpage.",
                func=lambda url, search_terms: web_crawler.run("search", url, search_terms=search_terms),
            )
        )
        
        self._register_tool(
            Tool(
                name="extract_webpage_elements",
                description="Extract specific elements from a webpage using CSS selectors.",
                func=lambda url, selectors: web_crawler.run("extract", url, selectors=selectors),
            )
        )
    
    def _register_tool(self, tool: BaseTool):
        """
        Register a tool with the manager.
        
        Args:
            tool: The tool to register
        """
        self.available_tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The tool if available, None otherwise
        """
        tool = self.available_tools.get(name)
        if not tool:
            logger.warning(f"Tool not found: {name}")
        return tool
    
    def get_tools_for_agent(self, agent_type: str) -> List[BaseTool]:
        """
        Get all tools available for a specific agent type.
        
        Args:
            agent_type: Type of agent (RESEARCHER, CODER, etc.)
            
        Returns:
            List of tools for the agent
        """
        if agent_type == "RESEARCHER":
            # Researcher gets search and crawler tools
            return [
                self.available_tools["brave_search"],
                self.available_tools["get_webpage_content"],
                self.available_tools["follow_webpage_links"], 
                self.available_tools["search_webpage_content"],
            ]
        elif agent_type == "CODER":
            # Coder might need to look up documentation
            return [
                self.available_tools["brave_search"],
                self.available_tools["get_webpage_content"],
            ]
        elif agent_type == "BROWSER":
            # Browser agent needs crawler tools
            return [
                self.available_tools["get_webpage_content"],
                self.available_tools["follow_webpage_links"],
                self.available_tools["search_webpage_content"],
                self.available_tools["extract_webpage_elements"],
            ]
        else:
            # Default minimal set
            return [self.available_tools["brave_search"]]
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all available tools.
        
        Returns:
            List of all registered tools
        """
        return list(self.available_tools.values())

# Create a global instance
tool_manager = ToolManager()

def get_tools_for_agent(agent_type: str) -> List[BaseTool]:
    """
    Get the appropriate tools for a specific agent type.
    
    Args:
        agent_type: Type of agent (RESEARCHER, CODER, etc.)
        
    Returns:
        List of tools for the agent
    """
    return tool_manager.get_tools_for_agent(agent_type) 
import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS, BRAVE_API_KEY, BRAVE_MAX_RESULTS
from .decorators import create_logged_tool
from .brave_search import BraveSearchTool

logger = logging.getLogger(__name__)

# Initialize Brave search tool
brave_tool = None
if BRAVE_API_KEY:
    brave_tool = BraveSearchTool(
        name="brave_search",
        api_key=BRAVE_API_KEY,
        max_results=BRAVE_MAX_RESULTS
    )
    logger.info("Brave Search initialized successfully")
else:
    logger.warning("Brave Search API key not found. Search functionality will be limited.")

# Initialize Tavily search tool with logging (only as fallback)
tavily_tool = None
try:
    LoggedTavilySearch = create_logged_tool(TavilySearchResults)
    tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
    logger.info("Tavily Search initialized as fallback")
except Exception as e:
    logger.warning(f"Failed to initialize Tavily Search: {e}")

# Use Brave search by default if available, otherwise try to fall back to Tavily
default_search_tool = brave_tool if brave_tool else tavily_tool

# If no search tool is available, raise a clear warning
if default_search_tool is None:
    logger.error("No search tool available. Please provide at least one valid search API key (Brave Search preferred).")

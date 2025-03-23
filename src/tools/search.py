import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS, BRAVE_API_KEY, BRAVE_MAX_RESULTS
from .decorators import create_logged_tool
from .brave_search import BraveSearchTool

logger = logging.getLogger(__name__)

# Initialize Brave search tool
brave_tool = None
if BRAVE_API_KEY:
    try:
        logger.debug(f"Attempting to initialize Brave Search with API key: {BRAVE_API_KEY[:5]}...")
        brave_tool = BraveSearchTool(
            name="brave_search",
            api_key=BRAVE_API_KEY,
            max_results=BRAVE_MAX_RESULTS
        )
        # Test the search tool with a meaningful query that should return reliable results
        try:
            logger.debug("Testing Brave Search with a specific query...")
            test_query = "artificial intelligence latest developments"  # More likely to return relevant results
            test_result = brave_tool.invoke(test_query)
            
            # Validate the result structure and content
            if isinstance(test_result, list) and len(test_result) > 0:
                # Check if at least one result contains relevant keywords
                relevant_keywords = ["ai", "artificial intelligence", "machine learning", "neural", "deep learning"]
                has_relevant_result = False
                
                for item in test_result:
                    # Convert item to string if it's a dictionary
                    item_text = str(item).lower()
                    if any(keyword in item_text for keyword in relevant_keywords):
                        has_relevant_result = True
                        break
                
                if has_relevant_result:
                    logger.info("Brave Search initialized and tested successfully")
                else:
                    logger.warning("Brave Search returned results but they may not be relevant")
                    # Still consider it working, as relevance is subjective
            elif isinstance(test_result, str) and len(test_result.strip()) > 100:
                # For string results, just check for minimal length
                logger.info("Brave Search initialized and returned valid string response")
            else:
                logger.warning(f"Brave Search initialized but returned unexpected result format: {type(test_result)}")
                logger.warning("Disabling Brave Search due to unexpected result format")
                brave_tool = None
        except Exception as test_error:
            logger.warning(f"Brave Search initialized but test query failed: {test_error}")
            logger.warning("Disabling Brave Search due to failed test")
            brave_tool = None
    except Exception as e:
        logger.error(f"Error initializing Brave Search despite having API key: {e}")
        brave_tool = None  # Ensure it's None if initialization failed
else:
    logger.warning("Brave Search API key not found. Search functionality will use fallback.")

# Initialize Tavily search tool ONLY if Brave is not available
tavily_tool = None
if brave_tool is None:
    try:
        logger.debug("Brave Search unavailable, attempting to initialize Tavily...")
        LoggedTavilySearch = create_logged_tool(TavilySearchResults)
        tavily_tool = LoggedTavilySearch(name="tavily_search", max_results=TAVILY_MAX_RESULTS)
        
        # Test the Tavily tool with the same query for consistency
        try:
            test_query = "artificial intelligence latest developments"
            test_result = tavily_tool.invoke(test_query)
            if test_result and isinstance(test_result, str) and len(test_result.strip()) > 100:
                logger.info("Tavily Search initialized and tested successfully as fallback")
            else:
                logger.warning(f"Tavily Search initialized but returned unexpected result: {test_result[:100]}...")
        except Exception as test_error:
            logger.warning(f"Tavily Search initialized but test query failed: {test_error}")
            # Still keep Tavily as fallback even if test fails
    except Exception as e:
        logger.warning(f"Failed to initialize Tavily Search fallback: {e}")
else:
    logger.debug("Skipping Tavily initialization since Brave Search is available")

# Use Brave search by default if available, otherwise try to fall back to Tavily
default_search_tool = brave_tool if brave_tool else tavily_tool

# Log diagnostic information about the available search tools
logger.debug(f"Search tool status - Brave: {'Available' if brave_tool else 'Unavailable'}, " +
             f"Tavily: {'Available' if tavily_tool else 'Unavailable'}, " +
             f"Default: {default_search_tool.__class__.__name__ if default_search_tool else 'None'}")

# If no search tool is available, raise a clear warning
if default_search_tool is None:
    logger.error("No search tool available. Please provide at least one valid search API key (Brave Search preferred).")
    
# Define a function to get the best available search tool
def get_best_search_tool():
    """
    Returns the best available search tool, with proper fallback logic.
    
    Returns:
        The best available search tool or None if none are available
    """
    if brave_tool is not None:
        return brave_tool
    elif tavily_tool is not None:
        logger.warning("Using Tavily as fallback search tool")
        return tavily_tool
    else:
        logger.error("No search tools available - search functionality will be severely limited")
        return None

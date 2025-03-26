import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import TAVILY_MAX_RESULTS, BRAVE_API_KEY, BRAVE_MAX_RESULTS
from .decorators import create_logged_tool
from .brave_search import BraveSearchTool
import os
import json
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from requests.exceptions import RequestException
from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# Try to get API key from multiple sources for better reliability
BRAVE_SEARCH_API_KEY = BRAVE_API_KEY or os.environ.get("BRAVE_API_KEY")

# Log API key status for debugging
if BRAVE_SEARCH_API_KEY:
    logger.info("Brave Search API key found.")
else:
    logger.warning("No Brave Search API key found. Search functionality will be limited.")
    # Check environment variable directly
    logger.debug(f"Direct environment check: BRAVE_API_KEY = {'Set' if os.environ.get('BRAVE_API_KEY') else 'Not set'}")

class BraveSearch:
    """
    Implementation of Brave Search API for web search capabilities.
    
    Brave Search provides independent search results with privacy protection
    and no reliance on Google or Bing indexes.
    """
    
    def __init__(self, api_key: Optional[str] = None, max_results: int = 5):
        """
        Initialize the Brave Search tool.
        
        Args:
            api_key: Brave Search API key (defaults to environment variable)
            max_results: Maximum number of results to return
        """
        self.api_key = api_key or BRAVE_SEARCH_API_KEY
        self.max_results = max_results
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
        if not self.api_key:
            logger.warning("No Brave Search API key found. Search functionality will be limited.")
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search using the Brave Search API.
        
        Args:
            query: The search query
            
        Returns:
            List of search results with title, URL, and snippet
        """
        if not self.api_key:
            logger.error("Brave Search API key not configured")
            return self._fallback_search_response(query)
        
        try:
            # Format the request to Brave Search API
            encoded_query = quote(query)
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }
            
            params = {
                "q": query,
                "count": self.max_results,
                "search_lang": "en",
                "safesearch": "moderate"
            }
            
            # Make the request
            response = requests.get(
                self.base_url,
                headers=headers,
                params=params
            )
            
            # Handle response
            if response.status_code == 200:
                return self._parse_brave_response(response.json(), query)
            else:
                logger.error(f"Brave Search API error: {response.status_code}, {response.text}")
                return self._fallback_search_response(query)
                
        except Exception as e:
            logger.error(f"Error in Brave search: {str(e)}")
            return self._fallback_search_response(query)
    
    def _parse_brave_response(self, response_data: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        Parse the Brave Search API response into a standardized format.
        
        Args:
            response_data: Raw API response data
            query: Original search query
            
        Returns:
            List of standardized search results
        """
        results = []
        
        try:
            # Check if we have web results
            if "web" in response_data and "results" in response_data["web"]:
                web_results = response_data["web"]["results"]
                
                for result in web_results[:self.max_results]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", ""),
                        "source": "Brave Search"
                    })
            
            # If no results, add a note
            if not results:
                logger.warning(f"No Brave Search results found for: {query}")
                results.append({
                    "title": "No Results Found",
                    "url": "",
                    "snippet": f"No search results were found for your query: {query}",
                    "source": "Brave Search"
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error parsing Brave Search response: {str(e)}")
            return self._fallback_search_response(query)
    
    def _fallback_search_response(self, query: str) -> List[Dict[str, Any]]:
        """
        Generate a fallback response when search fails.
        
        Args:
            query: Original search query
            
        Returns:
            Fallback search results
        """
        return [{
            "title": "Search Unavailable",
            "url": "",
            "snippet": (
                f"Unable to search for '{query}'. Search functionality is currently "
                "unavailable. Please check your Brave Search API key configuration."
            ),
            "source": "Fallback Response"
        }]
    
    def run(self, query: str) -> str:
        """
        Run the search and return results as a formatted string.
        Compatible with LangChain tool interface.
        
        Args:
            query: The search query
            
        Returns:
            Formatted string of search results
        """
        results = self.search(query)
        
        # Format the results as a string
        formatted_results = [
            f"Title: {result['title']}\nURL: {result['url']}\nSnippet: {result['snippet']}\n"
            for result in results
        ]
        
        return "\n".join(formatted_results)

# Create a global instance
brave_search = BraveSearch()

def search(query: str) -> List[Dict[str, Any]]:
    """
    Perform a web search using Brave Search.
    
    Args:
        query: Search query
        
    Returns:
        List of search results
    """
    return brave_search.search(query)

# Initialize Brave search tool
brave_tool = None
if BRAVE_SEARCH_API_KEY:
    brave_tool = BraveSearchTool(api_key=BRAVE_SEARCH_API_KEY, max_results=5)
else:
    logger.warning("Brave Search API key not configured. Web search functionality will be limited.")

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

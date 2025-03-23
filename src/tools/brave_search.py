import requests
import logging
import json
from typing import Dict, List, Optional, Union
from langchain.tools.base import BaseTool

logger = logging.getLogger(__name__)

class BraveSearchTool(BaseTool):
    """Tool that performs Brave Search API calls."""

    name: str = "brave_search"
    description: str = "Search for information using Brave Search API."
    api_key: str
    max_results: int = 5
    rate_limit_pause: float = 1.0  # Pause between requests to respect 1 req/sec limit

    def _run(self, query: str) -> List[Dict[str, str]]:
        """Run Brave search with the provided query."""
        logger.info(f"Performing Brave search for: {query}")
        
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        
        params = {
            "q": query,
            "count": self.max_results
        }
        
        try:
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            results = response.json()
            
            if "web" not in results or "results" not in results["web"]:
                logger.warning(f"Brave search returned unexpected format: {results}")
                return []
            
            formatted_results = []
            for item in results["web"]["results"][:self.max_results]:
                formatted_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", "")
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Brave search failed: {str(e)}")
            return [{
                "title": "Error",
                "url": "",
                "description": f"Search failed: {str(e)}"
            }]

    async def _arun(self, query: str) -> List[Dict[str, str]]:
        """Run Brave search asynchronously with query."""
        # This is a simple wrapper around _run
        # In a real implementation, you'd use aiohttp for async requests
        return self._run(query) 
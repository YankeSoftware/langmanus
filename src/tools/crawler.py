import os
import logging
import requests
from typing import Optional, Dict, Any, List, Union
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
import time

logger = logging.getLogger(__name__)

class WebCrawler:
    """
    A web crawler that can extract content from websites.
    Provides capabilities for content extraction, link following, and basic interaction.
    """
    
    def __init__(self):
        """Initialize the web crawler with default settings."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        self.timeout = 10  # Request timeout in seconds
        self.max_retries = 2
        self.retry_delay = 1  # Delay between retries in seconds
        self.max_links_to_follow = 10
        self.visited_urls = set()
    
    def get_page_content(self, url: str) -> Dict[str, Any]:
        """
        Fetch and process a web page's content.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Dictionary containing processed page content
        """
        logger.info(f"Fetching URL: {url}")
        
        # Check if URL has been visited
        if url in self.visited_urls:
            logger.info(f"URL already visited: {url}")
            return {
                "success": False,
                "url": url,
                "content": "",
                "error": "URL already visited",
                "links": [],
                "title": ""
            }
        
        # Add URL to visited set
        self.visited_urls.add(url)
        
        # Attempt to fetch the page with retries
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                # Check if successful
                if response.status_code == 200:
                    # Parse the content
                    return self._parse_page_content(response, url)
                else:
                    logger.warning(f"Request failed with status code: {response.status_code}")
                    if attempt < self.max_retries:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                    else:
                        return {
                            "success": False,
                            "url": url,
                            "content": "",
                            "error": f"Request failed with status code: {response.status_code}",
                            "links": [],
                            "title": ""
                        }
            
            except Exception as e:
                logger.error(f"Error fetching URL {url}: {str(e)}")
                if attempt < self.max_retries:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    return {
                        "success": False,
                        "url": url,
                        "content": "",
                        "error": f"Error: {str(e)}",
                        "links": [],
                        "title": ""
                    }
    
    def _parse_page_content(self, response: requests.Response, url: str) -> Dict[str, Any]:
        """
        Parse the content of a web page using BeautifulSoup.
        
        Args:
            response: The HTTP response
            url: The URL of the page
            
        Returns:
            Dictionary with parsed content
        """
        try:
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract page title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.text.strip()
            
            # Extract main content
            # Remove script, style, and other non-content elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Clean up extra whitespace
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href and not href.startswith(('#', 'javascript:', 'mailto:')):
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(url, href)
                    link_text = link.get_text(strip=True)
                    links.append({
                        "url": absolute_url,
                        "text": link_text if link_text else absolute_url
                    })
            
            # Limit number of links
            links = links[:self.max_links_to_follow]
            
            return {
                "success": True,
                "url": url,
                "content": text_content,
                "links": links,
                "title": title
            }
            
        except Exception as e:
            logger.error(f"Error parsing content from {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "content": "",
                "error": f"Parsing error: {str(e)}",
                "links": [],
                "title": ""
            }
    
    def extract_specific_content(self, url: str, selectors: List[str]) -> Dict[str, Any]:
        """
        Extract specific content from a page using CSS selectors.
        
        Args:
            url: The URL to fetch
            selectors: List of CSS selectors to extract
            
        Returns:
            Dictionary with extracted content
        """
        # Get the full page first
        page_result = self.get_page_content(url)
        
        if not page_result["success"]:
            return page_result
        
        try:
            # Re-fetch the content and apply selectors
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            extracted_content = {}
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    extracted_content[selector] = [element.get_text(strip=True) for element in elements]
                else:
                    extracted_content[selector] = []
            
            # Update the result with extracted content
            page_result["extracted_content"] = extracted_content
            
            return page_result
            
        except Exception as e:
            logger.error(f"Error extracting specific content from {url}: {str(e)}")
            return {
                "success": False,
                "url": url,
                "content": "",
                "error": f"Extraction error: {str(e)}",
                "links": [],
                "title": ""
            }
    
    def follow_links(self, start_url: str, depth: int = 1, max_pages: int = 3) -> List[Dict[str, Any]]:
        """
        Crawl a website starting from a URL, following links up to a certain depth.
        
        Args:
            start_url: Starting URL for crawling
            depth: How many levels deep to crawl
            max_pages: Maximum number of pages to visit
            
        Returns:
            List of dictionaries with page content
        """
        visited = set()
        results = []
        queue = [(start_url, 0)]  # (url, depth)
        
        while queue and len(results) < max_pages:
            url, current_depth = queue.pop(0)
            
            # Skip if already visited or exceeds depth
            if url in visited or current_depth > depth:
                continue
            
            # Mark as visited
            visited.add(url)
            
            # Fetch page content
            page_result = self.get_page_content(url)
            
            if page_result["success"]:
                # Add to results
                results.append(page_result)
                
                # Add links to queue if depth not reached
                if current_depth < depth:
                    for link in page_result["links"]:
                        queue.append((link["url"], current_depth + 1))
        
        return results
    
    def search_for_content(self, url: str, search_terms: List[str]) -> Dict[str, Any]:
        """
        Search for specific terms within a page's content.
        
        Args:
            url: The URL to search
            search_terms: List of terms to search for
            
        Returns:
            Dictionary with search results
        """
        page_result = self.get_page_content(url)
        
        if not page_result["success"]:
            return page_result
        
        content = page_result["content"].lower()
        
        search_results = {}
        for term in search_terms:
            term_lower = term.lower()
            if term_lower in content:
                # Find context around the term
                matches = re.finditer(re.escape(term_lower), content)
                contexts = []
                
                for match in matches:
                    start_pos = max(0, match.start() - 50)
                    end_pos = min(len(content), match.end() + 50)
                    context = content[start_pos:end_pos]
                    contexts.append(f"...{context}...")
                
                search_results[term] = {
                    "found": True,
                    "count": content.count(term_lower),
                    "contexts": contexts[:5]  # Limit to 5 contexts
                }
            else:
                search_results[term] = {
                    "found": False,
                    "count": 0,
                    "contexts": []
                }
        
        page_result["search_results"] = search_results
        return page_result
    
    def run(self, action: str, url: str, **kwargs) -> str:
        """
        Run a crawler action and return results as a formatted string.
        Compatible with LangChain tool interface.
        
        Args:
            action: The action to perform (get_content, follow_links, search, extract)
            url: The target URL
            **kwargs: Additional action-specific parameters
            
        Returns:
            Formatted string of results
        """
        if action == "get_content":
            result = self.get_page_content(url)
            return f"Title: {result['title']}\nURL: {url}\nSuccess: {result['success']}\n\nContent:\n{result['content'][:2000]}...\n\nLinks: {len(result['links'])} found"
            
        elif action == "follow_links":
            depth = kwargs.get("depth", 1)
            max_pages = kwargs.get("max_pages", 3)
            results = self.follow_links(url, depth, max_pages)
            
            summary = f"Crawled {len(results)} pages starting from {url}:\n\n"
            for idx, result in enumerate(results):
                summary += f"{idx+1}. {result['title']} - {result['url']}\n"
            
            return summary
            
        elif action == "search":
            search_terms = kwargs.get("search_terms", [])
            if not search_terms:
                return "Error: No search terms provided"
            
            result = self.search_for_content(url, search_terms)
            
            if not result["success"]:
                return f"Error searching {url}: {result.get('error', 'Unknown error')}"
            
            summary = f"Search results for {url}:\n\n"
            for term, data in result["search_results"].items():
                summary += f"Term: {term}\n"
                summary += f"Found: {data['found']}\n"
                summary += f"Occurrences: {data['count']}\n"
                
                if data["contexts"]:
                    summary += "Contexts:\n"
                    for context in data["contexts"]:
                        summary += f"- {context}\n"
                
                summary += "\n"
            
            return summary
            
        elif action == "extract":
            selectors = kwargs.get("selectors", [])
            if not selectors:
                return "Error: No CSS selectors provided"
            
            result = self.extract_specific_content(url, selectors)
            
            if not result["success"]:
                return f"Error extracting from {url}: {result.get('error', 'Unknown error')}"
            
            summary = f"Extracted content from {url}:\n\n"
            for selector, content in result.get("extracted_content", {}).items():
                summary += f"Selector: {selector}\n"
                summary += f"Results: {len(content)} found\n"
                
                for idx, item in enumerate(content[:3]):  # Limit to first 3 for brevity
                    summary += f"- {item}\n"
                
                if len(content) > 3:
                    summary += f"... and {len(content) - 3} more\n"
                
                summary += "\n"
            
            return summary
        
        else:
            return f"Error: Unknown action '{action}'. Valid actions are: get_content, follow_links, search, extract"

# Create a global instance
web_crawler = WebCrawler()

def crawl(action: str, url: str, **kwargs) -> str:
    """
    Interface for crawler tool to be used in agents.
    
    Args:
        action: The action to perform (get_content, follow_links, search, extract)
        url: The target URL
        **kwargs: Additional action-specific parameters
        
    Returns:
        Formatted results string
    """
    return web_crawler.run(action, url, **kwargs) 
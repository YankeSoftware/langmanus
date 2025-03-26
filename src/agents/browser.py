import logging
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.models import get_model_for_agent
from src.tools.tool_manager import get_tools_for_agent

logger = logging.getLogger(__name__)

class BrowserAgent:
    """
    Browser agent that simulates web interaction and browsing activities.
    This agent describes how to interact with web interfaces and navigate sites.
    
    Uses actual web crawler tools to provide accurate descriptions and information
    about websites rather than just simulating browsing.
    """
    
    def __init__(self):
        """Initialize the browser agent with models and templates."""
        # Get the appropriate model
        self.model = get_model_for_agent("BROWSER")
        
        # Get the browser-specific tools
        self.tools = get_tools_for_agent("BROWSER")
        
        # Create the system prompt template
        self.system_prompt = """You are an expert Browser Agent who specializes in web interactions and navigation.

Your role is to interact with web interfaces to accomplish specific tasks and extract useful information.

Follow these guidelines:

1. WEB INTERACTION APPROACH:
   - Provide detailed step-by-step instructions for web navigation
   - Use the crawler tools available to you to extract actual content from webpages
   - Follow links to explore related content when relevant
   - Search within webpages for specific information
   - Extract structured data from websites when possible

2. CONTENT EXTRACTION:
   - Use get_webpage_content to fetch the full text of a webpage
   - Use follow_webpage_links to explore related pages from a starting point
   - Use search_webpage_content to find specific information within a page
   - Use extract_webpage_elements to extract specific parts of a webpage
   - Provide concise summaries of the most relevant information found

3. WORKFLOW APPROACH:
   - Start by getting the main content of the most relevant page
   - Follow important links to get context or additional information
   - Search within pages for specific details relevant to the query
   - Extract only the most relevant content, avoiding information overload
   - Organize what you find in a clear, structured format

4. TECHNICAL CONSIDERATIONS:
   - Some websites may block or limit crawler access
   - Dynamic content (JavaScript-rendered) may not be fully accessible
   - Handle errors gracefully when pages can't be accessed
   - Avoid making too many requests to the same domain in rapid succession
   - Be selective about which links to follow to avoid going down rabbit holes

5. CONTENT ORGANIZATION:
   - Present findings in a clear, structured format
   - Group related information together
   - Highlight the most important or relevant details
   - Include source URLs for all information provided
   - Remove redundant or irrelevant information

Your primary goal is to provide information that is:
1. Accurate - based on actual webpage content
2. Relevant - focused on answering the user's specific question
3. Comprehensive - covering the important aspects of the topic
4. Well-organized - presented in a clear, logical structure
5. Actionable - enabling the user to take next steps if needed

Always conclude your response with:
1. A summary of the key information found
2. Any limitations or gaps in the information
3. A clear recommendation for what should happen next
"""
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and provide web interaction instructions.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Dict containing the browser instructions and routing metadata
        """
        try:
            # Extract the query and context
            query = ""
            context = []
            
            # First pass to get the original query
            for message in messages:
                if isinstance(message, dict) and message.get("type") == "human":
                    query = message.get("content", "")
                    break
                elif isinstance(message, HumanMessage):
                    query = message.content
                    break
            
            # Second pass to collect relevant context from other agents
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("metadata"), dict):
                    agent = message.get("metadata", {}).get("agent")
                    content = message.get("content", "")
                    
                    if agent and agent != "BROWSER" and content:
                        # Truncate long messages to keep context manageable
                        if len(content) > 1500:
                            truncated_content = content[:1500] + "... [content truncated]"
                            context.append(f"--- {agent} CONTRIBUTION ---\n{truncated_content}\n")
                        else:
                            context.append(f"--- {agent} CONTRIBUTION ---\n{content}\n")
            
            # Check if we have a valid query
            if not query:
                return {
                    "content": "I couldn't find a clear web interaction task to perform. Please provide a specific website or web task to navigate.",
                    "metadata": {"agent": "BROWSER", "next": "COORDINATOR"}
                }
            
            # Combine the context
            combined_context = "\n\n".join(context) if context else "No additional context available."
            
            # Initialize browsing state if not already present
            if "browsing" not in state:
                state["browsing"] = {
                    "sites_visited": [],
                    "tasks_performed": [],
                }
            
            # Create the prompt for web browsing
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", f"""Task: {query}

Additional Context:
{combined_context}

Please use your web crawler tools to gather relevant information for this task.
Extract content from relevant webpages, follow important links, and search for specific information when needed.
Present your findings in a clear, structured format that directly addresses the task.

Available tools:
- get_webpage_content: Fetches the full content of a webpage
- follow_webpage_links: Follows links from a starting URL to explore related content
- search_webpage_content: Searches for specific terms within a webpage
- extract_webpage_elements: Extracts specific elements from a webpage using CSS selectors
""")
            ])
            
            # Generate the browsing response
            response = self.model.invoke(prompt)
            
            # Extract the content
            browsing_content = response.content if hasattr(response, 'content') else str(response)
            
            # Update browsing state with the task
            state["browsing"]["tasks_performed"].append(query)
            
            # Extract mentioned URLs from the content using simple pattern matching
            import re
            url_pattern = r'https?://[^\s)"]+'
            mentioned_urls = re.findall(url_pattern, browsing_content)
            
            # Add extracted URLs to browsing state
            if mentioned_urls:
                state["browsing"]["sites_visited"].extend(mentioned_urls)
            
            # Determine next agent based on the browsing content
            next_agent = self._determine_next_agent(browsing_content)
            
            # Return the browsing instructions with metadata
            return {
                "content": browsing_content,
                "metadata": {"agent": "BROWSER", "next": next_agent}
            }
            
        except Exception as e:
            logger.error(f"Error in browser agent: {str(e)}")
            return {
                "content": f"I encountered an error while browsing: {str(e)}",
                "metadata": {"agent": "BROWSER", "next": "SUPERVISOR"}
            }
    
    def _determine_next_agent(self, browsing_content: str) -> str:
        """
        Determine which agent should handle the task next based on browsing content.
        
        Args:
            browsing_content: The content of the browsing
            
        Returns:
            The next agent to route to
        """
        lower_content = browsing_content.lower()
        
        # Look for explicit recommendations in the content
        if "next agent: researcher" in lower_content or "route to researcher" in lower_content:
            return "RESEARCHER"
        elif "next agent: planner" in lower_content or "route to planner" in lower_content:
            return "PLANNER"
        elif "next agent: coder" in lower_content or "route to coder" in lower_content:
            return "CODER"
        elif "next agent: reporter" in lower_content or "route to reporter" in lower_content:
            return "REPORTER"
        
        # If more research is specifically requested
        if "need more research" in lower_content or "additional information needed" in lower_content:
            return "RESEARCHER"
        
        # If code or implementation is mentioned
        if "implement" in lower_content or "code" in lower_content or "script" in lower_content:
            return "CODER"
        
        # Default to reporter to summarize findings
        return "REPORTER"

# Create the browser agent instance
browser_agent = BrowserAgent()

def browser_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming messages in the browser node.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the browser instructions and routing metadata
    """
    return browser_agent.process_messages(messages, state, metadata) 
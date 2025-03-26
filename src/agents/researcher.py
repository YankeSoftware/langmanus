import logging
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config.models import get_model_for_agent
from src.tools.tool_manager import get_tools_for_agent
from src.prompts.template import apply_prompt_template

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Research agent that gathers information from the web using Brave Search
    and crawling capabilities.
    
    Handles search failures gracefully and tracks research progress across invocations.
    """
    
    def __init__(self, api_config: Dict[str, Any] = None):
        """
        Initialize the research agent with search tools and appropriate model.
        
        Args:
            api_config: Optional configuration for API endpoints
        """
        # Get the appropriate model (local or DeepSeek)
        self.model = get_model_for_agent("RESEARCHER", api_config)
        
        # Get the research-specific tools from tool manager
        self.tools = get_tools_for_agent("RESEARCHER")
        
        # Create the prompt template
        prompt_template = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        
        # Create the agent
        self.agent = create_react_agent(
            llm=self.model,
            tools=self.tools,
            prompt=prompt_template
        )
        
        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="force",
        )
    
    def _synthesize_from_knowledge(self, query: str) -> str:
        """
        Fallback method when searches fail to provide reliable information.
        
        Args:
            query: The search query that failed to yield results
            
        Returns:
            A string with synthesized information from general knowledge
        """
        response = f"[KNOWLEDGE SYNTHESIS - NO SEARCH RESULTS AVAILABLE]\n\n"
        response += f"I couldn't find specific search results for '{query}'. Based on general knowledge:\n\n"
        
        # The LLM will generate a response here based on its training data
        # This is just a placeholder; the actual response will come from the model
        
        response += "\n\n[NOTE: This information is based on general knowledge rather than current search results. "
        response += "It may not reflect the most recent developments or specific details. "
        response += "Consider this as background information only.]"
        
        return response
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and perform research.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Agent's response with research findings
        """
        try:
            # Extract the query from the messages
            query = ""
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("type") == "human":
                    query = message.get("content", "")
                    break
                elif isinstance(message, dict) and message.get("role") == "user":
                    query = message.get("content", "")
                    break
                elif isinstance(message, HumanMessage):
                    query = message.content
                    break
            
            if not query:
                # Try to find any message with content
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get("content"):
                        query = message.get("content", "")
                        break
                    elif hasattr(message, "content") and message.content:
                        query = message.content
                        break
            
            if not query:
                logger.error("No query found in messages")
                return {
                    "content": "I couldn't find a clear query to research. Please provide a specific question or topic to investigate.",
                    "metadata": {"agent": "RESEARCHER", "next": "COORDINATOR"}
                }
            
            # Initialize research state if not already present
            if "research" not in state:
                state["research"] = {
                    "queries_attempted": [],
                    "successful_searches": [],
                    "failed_searches": [],
                    "topics_covered": [],
                    "findings": {},
                }
            
            # Check if this query has already been attempted
            if query in state["research"]["queries_attempted"]:
                # Try a variation of the query
                modified_query = f"latest information about {query}" if "latest" not in query else f"detailed explanation of {query}"
                logger.info(f"Query already attempted. Trying variation: {modified_query}")
                
                # Add to attempted queries
                state["research"]["queries_attempted"].append(modified_query)
                query = modified_query
            else:
                # Add to attempted queries
                state["research"]["queries_attempted"].append(query)
            
            # Apply the researcher prompt template from the prompts folder
            system_message = apply_prompt_template("researcher", state)

            # Create input for the agent with the system message first
            agent_input = {
                "messages": [system_message, HumanMessage(content=query)],
            }
            
            # Run the agent
            try:
                result = self.agent_executor.invoke(agent_input)
                
                # Record successful search
                state["research"]["successful_searches"].append(query)
                
                # Extract topics from the response
                topics = []
                lines = result["output"].split("\n")
                for line in lines:
                    if line.strip().startswith("#") or line.strip().startswith("*"):
                        topic = line.strip().strip("#").strip("*").strip()
                        if topic and len(topic) > 3:  # Avoid short/empty topics
                            topics.append(topic)
                
                # Add extracted topics
                if topics:
                    state["research"]["topics_covered"].extend(topics)
                
                # Return the result with metadata
                return {
                    "content": result["output"],
                    "metadata": {
                        "agent": "RESEARCHER",
                        "next": self._determine_next_step(result["output"], state)
                    }
                }
                
            except Exception as e:
                logger.error(f"Error in agent execution: {str(e)}")
                
                # Record failed search
                state["research"]["failed_searches"].append(query)
                
                # Fallback to knowledge synthesis
                fallback_response = self._synthesize_from_knowledge(query)
                
                return {
                    "content": f"I encountered an issue while researching '{query}': {str(e)}\n\n{fallback_response}",
                    "metadata": {"agent": "RESEARCHER", "next": "REPORTER"}
                }
                
        except Exception as e:
            logger.error(f"Error in research agent: {str(e)}")
            return {
                "content": f"I encountered an error while attempting to research: {str(e)}",
                "metadata": {"agent": "RESEARCHER", "next": "COORDINATOR"}
            }
    
    def _determine_next_step(self, research_output: str, state: Dict[str, Any]) -> str:
        """
        Determine the next agent to handle the task based on research output.
        
        Args:
            research_output: The output from the research
            state: Current state with research progress
            
        Returns:
            The next agent to route to
        """
        # Default to reporter if we've done several research iterations
        if len(state["research"]["successful_searches"]) >= 2:
            return "REPORTER"
        
        # Look for explicit routing in the output
        lower_output = research_output.lower()
        
        # Check for coding needs
        if "code" in lower_output or "implementation" in lower_output or "programming" in lower_output:
            return "CODER"
            
        # Check for planning needs
        if "plan" in lower_output or "strategy" in lower_output or "approach" in lower_output:
            return "PLANNER"
            
        # Check for browser needs
        if "browser" in lower_output or "interaction" in lower_output or "navigate" in lower_output:
            return "BROWSER"
            
        # Check for reporting readiness
        if "summary" in lower_output or "findings" in lower_output or "conclude" in lower_output:
            return "REPORTER"
            
        # Default to more research unless we've done enough
        search_count = len(state["research"]["successful_searches"])
        if search_count >= 3:
            return "REPORTER"  # Enough research, time to report
        else:
            return "RESEARCHER"  # Continue researching


# Create the research agent
research_agent = ResearchAgent()

def research_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming messages in the research node.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Agent's response with research findings
    """
    api_config = state.get("api_config", {})
    researcher = ResearchAgent(api_config)
    return researcher.process_messages(messages, state, metadata) 
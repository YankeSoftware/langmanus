import logging
import re
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.models import get_model_for_agent

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """
    Coordinator agent that orchestrates the workflow between different specialized agents.
    It serves as the entry point to the system and handles task delegation.
    """
    
    def __init__(self, api_config: Dict[str, Any] = None):
        """
        Initialize the coordinator agent with models and templates.
        
        Args:
            api_config: Optional configuration for API endpoints
        """
        # Get the appropriate model (local or DeepSeek)
        self.model = get_model_for_agent("COORDINATOR", api_config)
        
        # Create the system prompt template
        self.system_prompt = """You are an expert Coordinator Agent who orchestrates and manages a team of specialized AI agents.

Your primary responsibility is to analyze the user's query, understand what needs to be accomplished, and determine which specialized agent should handle the task next.

Follow these guidelines:

1. ANALYSIS APPROACH:
   - Begin by understanding the user's query thoroughly
   - Break complex tasks into logical subtasks
   - Identify the key domains and skills required (research, planning, coding, etc.)
   - Assess what information is needed vs. what is already available
   - Consider the dependencies between different parts of the task

2. WORKFLOW MANAGEMENT:
   - Start new inquiries with a research phase to gather information
   - Ensure logical progression through the workflow
   - Prevent loops where agents repeatedly hand off to each other without progress
   - Track which agents have already contributed and what they've provided
   - Make decisions based on the current state of the task, not just the last message

3. AGENT SELECTION CRITERIA:
   - RESEARCHER: For gathering information, answering factual questions, exploring topics
   - PLANNER: For developing strategies, outlining approaches, creating roadmaps
   - CODER: For implementing technical solutions, writing code, building systems
   - BROWSER: For interactive web tasks, navigation, form submission
   - REPORTER: For synthesizing information, creating final reports, presenting conclusions

4. COORDINATION PRIORITIES:
   - Maximize efficiency by minimizing unnecessary agent transitions
   - Ensure comprehensive coverage of all aspects of the user's query
   - Balance depth of investigation with timely completion
   - Adapt the workflow as new information emerges
   - Ensure the final output fully addresses the user's needs

5. DECISION MAKING:
   - Be decisive in your routing recommendations
   - Provide clear context for the next agent to understand their task
   - Include specific instructions for the next agent when needed
   - Make routing decisions based on what will advance the task most effectively
   - Consider whether enough information has been gathered before moving to solution phases

Always conclude your message with a clear routing directive in this format:
"ROUTE TO: [AGENT]" where [AGENT] is one of: RESEARCHER, PLANNER, CODER, BROWSER, or REPORTER.
"""
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the incoming messages and determine which agent to route to next.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Dict containing the coordinator's analysis and routing recommendation
        """
        try:
            # Extract the query
            query = ""
            for message in messages:
                if isinstance(message, dict) and message.get("type") == "human":
                    query = message.get("content", "")
                    break
                elif isinstance(message, HumanMessage):
                    query = message.content
                    break
            
            if not query:
                return {
                    "content": "I couldn't find a clear query to coordinate. Please provide a specific question or task.",
                    "metadata": {"agent": "COORDINATOR", "next": "FINISH"}
                }
            
            # Initialize workflow tracking if not already present
            if "workflow" not in state:
                state["workflow"] = {
                    "initiated_at": None,
                    "task_type": None,
                    "steps_completed": [],
                    "current_phase": "initiation",
                    "priority_agents": [],
                }
                
                # Set initiation time
                from datetime import datetime
                state["workflow"]["initiated_at"] = datetime.now().isoformat()
            
            # Check for contributing agents (initialized in builder.py)
            contributing_agents = state.get("contributing_agents", [])
            
            # Prepare context about the current state
            workflow_context = ""
            if contributing_agents:
                workflow_context += f"Agents that have already contributed: {', '.join(contributing_agents)}\n\n"
            
            # Identify the type of task if not already identified
            if not state["workflow"]["task_type"]:
                # Simple heuristic for determining task type
                lowercase_query = query.lower()
                
                if any(term in lowercase_query for term in ["research", "information", "learn about", "find out", "details on"]):
                    state["workflow"]["task_type"] = "research_focused"
                    state["workflow"]["priority_agents"] = ["RESEARCHER", "REPORTER"]
                    
                elif any(term in lowercase_query for term in ["build", "create", "develop", "implement", "code"]):
                    state["workflow"]["task_type"] = "development_focused"
                    state["workflow"]["priority_agents"] = ["PLANNER", "CODER", "REPORTER"]
                    
                elif any(term in lowercase_query for term in ["plan", "strategy", "approach", "method"]):
                    state["workflow"]["task_type"] = "planning_focused"
                    state["workflow"]["priority_agents"] = ["PLANNER", "RESEARCHER", "REPORTER"]
                    
                elif any(term in lowercase_query for term in ["browse", "website", "navigate", "web"]):
                    state["workflow"]["task_type"] = "browsing_focused"
                    state["workflow"]["priority_agents"] = ["BROWSER", "REPORTER"]
                    
                else:
                    state["workflow"]["task_type"] = "general_inquiry"
                    state["workflow"]["priority_agents"] = ["RESEARCHER", "PLANNER", "REPORTER"]
                
                workflow_context += f"Task type: {state['workflow']['task_type']}\n"
                workflow_context += f"Priority agents: {', '.join(state['workflow']['priority_agents'])}\n\n"
            
            # Create the prompt for the coordinator
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", f"""Original Query: {query}

Current Workflow State:
{workflow_context}

Please analyze this query and determine which specialized agent should handle it next.
Consider what information needs to be gathered, what planning might be needed, and what implementation steps could be required.
Based on your analysis, provide a clear routing directive to the appropriate agent.
""")
            ])
            
            # Generate the coordination response
            response = self.model.invoke(prompt)
            
            # Extract the content
            coordination_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the routing directive
            routing_pattern = r"ROUTE TO: (\w+)"
            match = re.search(routing_pattern, coordination_content, re.IGNORECASE)
            
            next_agent = "RESEARCHER"  # Default to researcher if no directive found
            
            if match:
                found_agent = match.group(1).upper()
                valid_agents = ["RESEARCHER", "PLANNER", "CODER", "BROWSER", "REPORTER"]
                
                if found_agent in valid_agents:
                    next_agent = found_agent
                else:
                    # Try to normalize the agent name
                    agent_map = {
                        "RESEARCH": "RESEARCHER",
                        "PLAN": "PLANNER",
                        "CODE": "CODER",
                        "BROWSE": "BROWSER",
                        "REPORT": "REPORTER"
                    }
                    
                    for key, value in agent_map.items():
                        if key in found_agent:
                            next_agent = value
                            break
            
            # Check for loop prevention - avoid going to the same agent repeatedly
            if "agent_visit_counts" in state:
                researcher_visits = state["agent_visit_counts"].get("RESEARCHER", 0)
                
                # If researcher has been visited too many times, route to planner or reporter
                if next_agent == "RESEARCHER" and researcher_visits >= 2:
                    logger.warning(f"Preventing research loop (visits: {researcher_visits}), routing to alternative")
                    
                    # Check if we have contributing agents to determine best alternative
                    if "PLANNER" not in contributing_agents:
                        next_agent = "PLANNER"
                        logger.info("Routing to PLANNER as RESEARCHER alternative (not yet contributed)")
                    else:
                        next_agent = "REPORTER"
                        logger.info("Routing to REPORTER to summarize findings and break loop")
            
            # Return the coordination response with routing metadata
            return {
                "content": coordination_content,
                "metadata": {"agent": "COORDINATOR", "next": next_agent}
            }
            
        except Exception as e:
            logger.error(f"Error in coordinator agent: {str(e)}")
            return {
                "content": f"I encountered an error while analyzing your query: {str(e)}. Let's start with research to understand more about your request.",
                "metadata": {"agent": "COORDINATOR", "next": "RESEARCHER"}
            }

# Create the coordinator agent instance
coordinator_agent = CoordinatorAgent()

def coordinator_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming messages in the coordinator node.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the coordination analysis and routing recommendation
    """
    return coordinator_agent.process_messages(messages, state, metadata) 
from langgraph.graph import StateGraph, START, END
from .types import State, Command
from .nodes import (
    supervisor_node,
    research_node,
    code_node,
    coordinator_node,
    browser_node,
    reporter_node,
    planner_node,
)
from src.config import TEAM_MEMBERS
import logging
import re

logger = logging.getLogger(__name__)


def build_graph():
    """Build and return the agent workflow graph."""
    # Create a new graph
    builder = StateGraph(State)
    
    # Define wrapper functions to ensure correct parameters are passed
    def coordinator_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return coordinator_node(state.get("messages", []), state, {})
    
    def planner_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return planner_node(state.get("messages", []), state, {})
    
    def supervisor_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return supervisor_node(state.get("messages", []), state, {})
    
    def research_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return research_node(state.get("messages", []), state, {})
    
    def code_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return code_node(state.get("messages", []), state, {})
    
    def browser_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return browser_node(state.get("messages", []), state, {})
    
    def reporter_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        return reporter_node(state.get("messages", []), state, {})
    
    # Add all nodes to the graph with wrapper functions
    builder.add_node("coordinator", coordinator_wrapper)
    builder.add_node("planner", planner_wrapper)
    builder.add_node("supervisor", supervisor_wrapper)
    builder.add_node("researcher", research_wrapper)
    builder.add_node("coder", code_wrapper)
    builder.add_node("browser", browser_wrapper)
    builder.add_node("reporter", reporter_wrapper)
    
    # Define entry point - workflow starts with the coordinator
    builder.add_edge(START, "coordinator")
    
    # Define the conditional edge from supervisor to other nodes
    def route_from_supervisor(state):
        """
        Route to the next node based on supervisor's decision.
        
        This function:
        1. Extracts the routing directive from the supervisor's output
        2. Validates the target node is valid and exists
        3. Provides extensive logging for debugging
        4. Handles edge cases and errors gracefully
        5. Enforces required agent contributions before FINISH
        
        Returns:
            String representing the next node to route to
        """
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
            logger.warning("Messages list was not initialized in state")
            logger.info("Routing to researcher as fallback for empty messages")
            return "researcher"  # Default to researcher instead of ending
            
        # Track contributing agents for workflow enforcement
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
            logger.info("Initializing contributing_agents tracking")
        
        # Log all messages for debugging
        logger.debug(f"Routing with {len(state.get('messages', []))} messages")
        logger.debug(f"Current contributing agents: {state.get('contributing_agents', [])}")
        
        # Get the last message with metadata
        routing_target = None
        previous_agent = None
        
        # First try to find a supervisor message with valid metadata
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, dict):
                # Track the previous agent for workflow enforcement
                if msg.get("metadata", {}).get("agent") and not previous_agent:
                    previous_agent = msg.get("metadata", {}).get("agent")
                    logger.debug(f"Previous agent: {previous_agent}")
                    
                    # Add to contributing agents if not already there
                    if previous_agent and previous_agent not in state.get("contributing_agents", []):
                        state["contributing_agents"] = state.get("contributing_agents", []) + [previous_agent]
                        logger.info(f"Added {previous_agent} to contributing agents")
                
                # Check for explicit routing in metadata
                if msg.get("metadata", {}).get("next"):
                    routing_target = msg.get("metadata", {}).get("next")
                    logger.info(f"Found routing target in metadata: {routing_target}")
                    break
                    
                # Check for SUPERVISOR messages to extract routing
                if "SUPERVISOR" in msg.get("content", "") or msg.get("metadata", {}).get("agent") == "SUPERVISOR":
                    logger.debug(f"Found supervisor message: {msg.get('content', '')[:100]}...")
                    
                    # Try to extract routing from content using regex
                    content = msg.get("content", "")
                    
                    # Try different patterns to extract routing target
                    patterns = [
                        r'route to[:\s]+([A-Z]+)',  # "route to X" pattern
                        r'next[:\s]+([A-Z]+)',  # "next: X" pattern
                        r'should route to[:\s]+([A-Z]+)',  # "should route to X" pattern
                        r'([A-Z]{5,})',  # Standalone words in uppercase with min 5 chars
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            routing_target = matches[0].strip().upper()
                            logger.info(f"Extracted routing target from content using pattern '{pattern}': {routing_target}")
                            break
                            
                    # If we found a routing target, break out of the loop
                    if routing_target:
                        break
        
        # If no routing found, use RESEARCHER as default
        if not routing_target:
            logger.warning("No routing target found in supervisor messages, defaulting to RESEARCHER")
            routing_target = "RESEARCHER"
        
        # Normalize the routing target
        routing_map = {
            'RESEARCH': 'RESEARCHER',
            'RESEARCH AGENT': 'RESEARCHER',
            'RESEAR': 'RESEARCHER',
            'PLAN': 'PLANNER',
            'PLANNING': 'PLANNER',
            'CODE': 'CODER',
            'CODING': 'CODER',
            'REPORT': 'REPORTER',
            'REPORTING': 'REPORTER',
            'END': 'FINISH',
            'DONE': 'FINISH',
            'COMPLETE': 'FINISH'
        }
        
        # Normalize target if it maps to a known variation
        if routing_target in routing_map:
            original_target = routing_target
            routing_target = routing_map[routing_target]
            logger.info(f"Normalized routing target from {original_target} to {routing_target}")
            
        # WORKFLOW ENFORCEMENT
        
        # Special case - if coming from coordinator, always go to researcher first
        if previous_agent == "COORDINATOR" and "RESEARCHER" not in state.get("contributing_agents", []):
            logger.info("Coming from coordinator without researcher contribution - forcing route to RESEARCHER")
            return "researcher"
        
        # Process the final routing decision
        if routing_target == "FINISH":
            # Check if researcher and reporter have contributed before allowing FINISH
            researcher_contributed = "RESEARCHER" in state.get("contributing_agents", [])
            reporter_contributed = "REPORTER" in state.get("contributing_agents", [])
            
            if not researcher_contributed:
                logger.warning("Attempted to FINISH but RESEARCHER hasn't contributed - routing to researcher")
                return "researcher"
            
            if not reporter_contributed:
                logger.warning("Attempted to FINISH but REPORTER hasn't contributed - routing to reporter")
                return "reporter"
            
            logger.info("FINISH is valid - both RESEARCHER and REPORTER have contributed")
            return "__end__"
        
        # Map the routing target to the node name
        node_mapping = {
            "COORDINATOR": "coordinator",
            "PLANNER": "planner", 
            "RESEARCHER": "researcher",
            "CODER": "coder",
            "BROWSER": "browser",
            "REPORTER": "reporter"
        }
        
        # Check if the routing target is valid
        if routing_target in node_mapping:
            node_name = node_mapping[routing_target]
            logger.info(f"Routing to {node_name}")
            return node_name
        
        # If we get here, the routing target wasn't valid
        logger.error(f"Invalid routing target: {routing_target}")
        logger.info("Routing to researcher as fallback for invalid target")
        return "researcher"
    
    # Add edges from each node to supervisor
    builder.add_edge("coordinator", "supervisor")
    builder.add_edge("planner", "supervisor")
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")
    builder.add_edge("browser", "supervisor")
    builder.add_edge("reporter", "supervisor")
    
    # Add conditional edge from supervisor to other nodes
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "coordinator": "coordinator",
            "planner": "planner",
            "researcher": "researcher",
            "coder": "coder", 
            "browser": "browser",
            "reporter": "reporter",
            "__end__": END,
        }
    )
    
    # Compile and return the graph
    return builder.compile()

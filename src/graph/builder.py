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
        """Route to the next node based on supervisor's decision."""
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
            logger.warning("Messages list was not initialized in state")
            return "__end__"
            
        # Get the last message with metadata
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, dict) and msg.get("metadata") and "next" in msg.get("metadata", {}):
                next_node = msg["metadata"]["next"]
                logger.info(f"Routing from supervisor to: {next_node}")
                
                # If FINISH is specified, end the workflow
                if next_node == "FINISH":
                    return "__end__"
                
                # Otherwise route to the specified team member (case-insensitive)
                for valid_node in TEAM_MEMBERS:
                    if valid_node.lower() == next_node.lower():
                        return valid_node.lower()
                
                # If we get here, the next_node wasn't found in team members
                logger.warning(f"Invalid routing target: {next_node}, ending workflow")
                return "__end__"
                
        # Default to ending if no valid routing found
        logger.warning("No valid routing found in supervisor output, ending workflow")
        return "__end__"
    
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

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
from typing import Dict

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
    def route_supervisor_decision(state: Dict):
        """Route the next action based on supervisor's decision."""
        logger.debug("Determining next action from supervisor...")
        
        # Get the last message
        if not state.get("messages") or len(state["messages"]) == 0:
            logger.warning("No messages in state to determine routing. Defaulting to FINISH.")
            return "__end__"
        
        last_msg = state["messages"][-1]
        
        # Try to extract routing from metadata first (preferred method)
        if isinstance(last_msg, dict) and "metadata" in last_msg:
            metadata = last_msg.get("metadata", {})
            if metadata and "next" in metadata:
                next_action = metadata["next"]
                logger.info(f"Routing from supervisor to: {next_action}")
                # If FINISH is specified, end the workflow
                if next_action == "FINISH":
                    return "__end__"
                # Make sure next_action is lowercase to match node names
                return next_action.lower()
        
        # As a fallback, look for routing in the content
        if isinstance(last_msg, dict) and "content" in last_msg:
            content = last_msg["content"]
            if "Next: " in content:
                parts = content.split("Next: ", 1)
                next_part = parts[1].split("\n", 1)[0].strip()
                logger.info(f"Extracted routing from content: {next_part}")
                # If FINISH is specified, end the workflow
                if next_part == "FINISH":
                    return "__end__"
                # Make sure next_part is lowercase to match node names
                return next_part.lower()
        
        # If we couldn't find a clear routing decision, check for FINISH indicators
        if isinstance(last_msg, dict) and "content" in last_msg:
            content = last_msg["content"].lower()
            if "error" in content or "finish" in content or "end" in content or "complete" in content:
                logger.info("Detected FINISH indicators in message content")
                return "__end__"
        
        logger.warning("Could not determine routing decision. Defaulting to FINISH.")
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
        route_supervisor_decision,
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

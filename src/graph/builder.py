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
import os
import time
import functools
import asyncio
from typing import Dict, Any, List, Callable, Optional
from functools import lru_cache
import traceback

logger = logging.getLogger(__name__)

# Check if LM Studio is being used
def is_lm_studio_model():
    """Check if LM Studio is being used as a model provider"""
    from src.config.env import LLM_TYPE, LLM_BASE_URL
    # Check if using local LM with default LM Studio port
    return (LLM_TYPE == "local" and LLM_BASE_URL and 
            ("localhost:1234" in LLM_BASE_URL or "127.0.0.1:1234" in LLM_BASE_URL))

# Performance optimization settings
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() in ("true", "1", "yes")
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
ENABLE_ASYNC = os.getenv("ENABLE_ASYNC", "true").lower() in ("true", "1", "yes")
BATCHING_ENABLED = os.getenv("ENABLE_BATCHING", "true").lower() in ("true", "1", "yes")

# Human in the Loop configuration
HITL_ENABLED = os.getenv("HITL_ENABLED", "true").lower() in ("true", "1", "yes")
HITL_APPROVAL_REQUIRED = os.getenv("HITL_APPROVAL_REQUIRED", "true").lower() in ("true", "1", "yes")

# Automatically enable HITL for LM Studio for compatibility
if is_lm_studio_model():
    logger.info("LM Studio detected - enabling HITL mode for compatibility")
    HITL_ENABLED = True
    
    # Log this important compatibility notice
    logger.warning("LM Studio has limited role support (only 'user' and 'assistant')")
    logger.warning("System messages are being converted to user messages for compatibility")

# Create a global cache for node results
# This reduces redundant API calls for identical inputs
node_result_cache = {}

def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    Decorator that creates a timed LRU cache for the decorated function.
    Cached results expire after the specified number of seconds.
    
    Args:
        seconds: Number of seconds to cache results
        maxsize: Maximum size of the cache
        
    Returns:
        Decorated function with timed caching
    """
    def decorator(func):
        # Create a cache dictionary instead of using lru_cache
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a hashable key from the arguments
            # For dictionaries and lists, convert them to their string representation
            key_parts = []
            
            # Process positional arguments
            for arg in args:
                if isinstance(arg, (dict, list)):
                    # Convert dict/list to a string representation
                    key_parts.append(str(arg))
                else:
                    key_parts.append(arg)
            
            # Process keyword arguments (sorted for consistency)
            for k in sorted(kwargs.keys()):
                v = kwargs[k]
                if isinstance(v, (dict, list)):
                    # Convert dict/list to a string representation
                    key_parts.append(f"{k}:{str(v)}")
                else:
                    key_parts.append(f"{k}:{v}")
            
            # Create a hashable key (using tuple since strings of large dicts can be very long)
            try:
                cache_key = hash(str(key_parts))
            except:
                # If hashing fails, use the function name as a key (fallback, less precise caching)
                cache_key = func.__name__
                
            current_time = time.time()
            
            # Check if the result is in the cache and not expired
            if cache_key in cache:
                result, timestamp = cache[cache_key]
                if current_time - timestamp <= seconds:
                    return result
            
            # Calculate the new result
            result = func(*args, **kwargs)
            
            # Store in cache
            cache[cache_key] = (result, current_time)
            
            # Limit cache size by removing oldest entries
            if len(cache) > maxsize:
                oldest_keys = sorted(cache.keys(), key=lambda k: cache[k][1])[:maxsize//5]
                for key in oldest_keys:
                    cache.pop(key, None)
                    
            return result

        return wrapper

    return decorator

def memoize_node(func):
    """
    Decorator to memoize node function results to reduce duplicate API calls.
    
    Args:
        func: Node function to memoize
        
    Returns:
        Memoized function
    """
    if not ENABLE_CACHING:
        return func
    
    # Create a cache for this specific function
    function_cache = {}
    
    @functools.wraps(func)
    def wrapper(state: Dict[str, Any], *args, **kwargs):
        try:
            # Extract the most relevant parts of state for caching
            # - Only use the last message content to avoid huge cache keys
            # - Avoid using mutable objects directly as keys
            cache_key_parts = []
            
            # Get the last message for caching
            messages = state.get("messages", [])
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    content = last_message.get("content", "")
                    if content:
                        # Use only first 100 chars of content for key
                        cache_key_parts.append(content[:100])
            
            # Also include function name in key
            cache_key_parts.append(func.__name__)
            
            # Generate a hash from the key parts
            cache_key = hash(str(cache_key_parts))
            
            current_time = time.time()
            
            # Check cache for existing result
            if cache_key in function_cache:
                result, timestamp = function_cache[cache_key]
                # If cache entry is still fresh (less than 5 minutes old)
                if current_time - timestamp < 300:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return result
            
            # Execute function if not in cache or cache stale
            logger.debug(f"Cache miss for {func.__name__}, executing")
            start_time = time.time()
            result = func(state, *args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Node {func.__name__} executed in {execution_time:.2f}s")
            
            # Cache the result
            function_cache[cache_key] = (result, current_time)
            
            # Cleanup old entries if cache is too large
            if len(function_cache) > CACHE_SIZE:
                # Remove oldest entries
                oldest_keys = sorted(function_cache.keys(), 
                                    key=lambda k: function_cache[k][1])[:CACHE_SIZE//5]
                for key in oldest_keys:
                    del function_cache[key]
                
            return result
        except Exception as e:
            # Log the error and fall back to uncached execution
            logger.error(f"Error in memoize_node for {func.__name__}: {str(e)}")
            logger.debug(f"Falling back to uncached execution for {func.__name__}")
            return func(state, *args, **kwargs)
            
    return wrapper

async def run_async_node(func, state):
    """Run a node function asynchronously."""
    return func(state)

def async_batch_executor(funcs, state):
    """
    Execute multiple node functions in parallel.
    
    Args:
        funcs: List of node functions to execute
        state: State to pass to each function
        
    Returns:
        List of results from each function
    """
    if not ENABLE_ASYNC:
        # Execute sequentially if async is disabled
        return [func(state) for func in funcs]
    
    # Create tasks for each function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [run_async_node(func, state) for func in funcs]
    
    # Run all tasks concurrently
    try:
        results = loop.run_until_complete(asyncio.gather(*tasks))
    finally:
        loop.close()
    
    return results

def build_graph():
    """Build and return the agent workflow graph."""
    # Create a new graph
    builder = StateGraph(State)
    
    # Define wrapper functions to ensure correct parameters are passed
    @memoize_node
    def coordinator_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = coordinator_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["coordinator"] = execution_time
        
        # Mark coordinator as having contributed
        if "contributing_agents" in state and "COORDINATOR" not in state["contributing_agents"]:
            state["contributing_agents"].append("COORDINATOR")
        return result
    
    @memoize_node
    def planner_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        # Check for HITL feedback if enabled
        if state.get("hitl_enabled", HITL_ENABLED) and state.get("hitl_feedback"):
            # Append HITL feedback to messages if not already there
            feedback_found = False
            for msg in state["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and "PLAN FEEDBACK:" in msg.get("content", ""):
                    feedback_found = True
                    break
            
            if not feedback_found and state["hitl_feedback"]:
                state["messages"].append({
                    "role": "user", 
                    "content": f"PLAN FEEDBACK: {state['hitl_feedback']}"
                })
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = planner_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["planner"] = execution_time
        
        # Mark planner as having contributed
        if "contributing_agents" in state and "PLANNER" not in state["contributing_agents"]:
            state["contributing_agents"].append("PLANNER")
        return result
    
    @memoize_node
    def supervisor_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = supervisor_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["supervisor"] = execution_time
        
        # Mark supervisor as having contributed
        if "contributing_agents" in state and "SUPERVISOR" not in state["contributing_agents"]:
            state["contributing_agents"].append("SUPERVISOR")
        return result
    
    @memoize_node
    def research_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = research_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["researcher"] = execution_time
        
        # Mark researcher as having contributed
        if "contributing_agents" in state and "RESEARCHER" not in state["contributing_agents"]:
            state["contributing_agents"].append("RESEARCHER")
        return result
    
    @memoize_node
    def code_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = code_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["coder"] = execution_time
        
        # Mark coder as having contributed
        if "contributing_agents" in state and "CODER" not in state["contributing_agents"]:
            state["contributing_agents"].append("CODER")
        return result
    
    @memoize_node
    def browser_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = browser_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["browser"] = execution_time
        
        # Mark browser as having contributed
        if "contributing_agents" in state and "BROWSER" not in state["contributing_agents"]:
            state["contributing_agents"].append("BROWSER")
        return result
    
    @memoize_node
    def reporter_wrapper(state):
        # Initialize messages list if it doesn't exist
        if "messages" not in state:
            state["messages"] = []
        # Track agent contributions
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Track execution metrics
        start_time = time.time()
        
        # Pass API config from state
        api_config = state.get("api_config", {})
        result = reporter_node(state.get("messages", []), state, {"api_config": api_config})
        
        # Record execution time
        execution_time = time.time() - start_time
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        state["execution_metrics"]["reporter"] = execution_time
        
        # Mark reporter as having contributed
        if "contributing_agents" in state and "REPORTER" not in state["contributing_agents"]:
            state["contributing_agents"].append("REPORTER")
        return result
    
    # Optimize with batch processing if enabled
    def initialize_state(state):
        """Initialize state with all required fields."""
        if "messages" not in state:
            state["messages"] = []
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        if "execution_metrics" not in state:
            state["execution_metrics"] = {}
        if "cache_hits" not in state:
            state["cache_hits"] = 0
        if "cache_misses" not in state:
            state["cache_misses"] = 0
        return state
    
    def batch_prepare_researcher_planner(state):
        """Batch prepare both researcher and planner nodes in parallel."""
        if not BATCHING_ENABLED:
            # Return unchanged state if batching is disabled
            return state
        
        try:
            # Initialize state
            initialize_state(state)
            
            # Get API config
            api_config = state.get("api_config", {})
            
            # Run researcher and planner in parallel for efficiency
            logger.debug("Starting batch execution of researcher and planner")
            start_time = time.time()
            
            # Create wrapped versions of the nodes for batch execution
            def batch_research():
                return research_node(state.get("messages", []), state, {"api_config": api_config})
            
            def batch_planning():
                return planner_node(state.get("messages", []), state, {"api_config": api_config})
            
            # Execute in parallel
            results = async_batch_executor([batch_research, batch_planning], state)
            
            # Process results (if parallel execution succeeded)
            if results and len(results) == 2:
                research_result, plan_result = results
                
                # Add research findings to messages
                if "messages" in state and research_result:
                    if isinstance(research_result, dict) and "content" in research_result:
                        state["messages"].append({
                            "role": "assistant",
                            "content": research_result.get("content", ""),
                            "metadata": research_result.get("metadata", {})
                        })
                
                # Add plan to messages
                if "messages" in state and plan_result:
                    if isinstance(plan_result, dict) and "content" in plan_result:
                        state["messages"].append({
                            "role": "assistant",
                            "content": plan_result.get("content", ""),
                            "metadata": plan_result.get("metadata", {})
                        })
                
                # Record execution metrics
                execution_time = time.time() - start_time
                if "execution_metrics" not in state:
                    state["execution_metrics"] = {}
                state["execution_metrics"]["batch_research_planning"] = execution_time
                logger.debug(f"Batch execution completed in {execution_time:.2f}s")
                
                # Mark agents as having contributed
                if "contributing_agents" not in state:
                    state["contributing_agents"] = []
                if "RESEARCHER" not in state["contributing_agents"]:
                    state["contributing_agents"].append("RESEARCHER")
                if "PLANNER" not in state["contributing_agents"]:
                    state["contributing_agents"].append("PLANNER")
            
            return state
        except Exception as e:
            logger.error(f"Error in batch execution: {str(e)}")
            logger.error(traceback.format_exc())
            # Return unchanged state on error
            return state
    
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
        try:
            # Validate state is a dictionary
            if not isinstance(state, dict):
                logger.error(f"State is not a dictionary: {type(state)}. Routing to coordinator.")
                return "coordinator"
            
            # Initialize messages list if it doesn't exist
            if "messages" not in state:
                state["messages"] = []
                logger.warning("Messages list was not initialized in state")
                logger.info("Routing to researcher as fallback for empty messages")
                return "researcher"  # Default to researcher instead of ending
            
            # Track contributing agents for workflow enforcement
            if "contributing_agents" not in state:
                state["contributing_agents"] = ["SUPERVISOR"]  # Start with supervisor
            
            # Get the last message from the supervisor
            messages = state.get("messages", [])
            if not messages:
                logger.warning("No messages in state. Routing to coordinator.")
                return "coordinator"
            
            # Find the latest supervisor message
            supervisor_message = None
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("metadata", {}).get("agent") == "SUPERVISOR":
                    supervisor_message = msg
                    break
                
            # If no supervisor message found, use the last message
            if not supervisor_message:
                supervisor_message = messages[-1]
            
            # Safety check for message format
            if not isinstance(supervisor_message, dict) or "content" not in supervisor_message:
                logger.warning("Last message is not a proper dict with content. Routing to coordinator.")
                return "coordinator"
            
            supervisor_content = supervisor_message.get("content", "").lower()
            
            # Log the content for debugging
            logger.debug(f"Routing based on supervisor content: {supervisor_content[:100]}...")
            
            # Check if routing is explicitly specified
            next_agent = None
            routing_keywords = {
                "route to: researcher": "researcher",
                "route to: planner": "planner",
                "route to: coordinator": "coordinator", 
                "route to: coder": "coder",
                "route to: browser": "browser",
                "route to: reporter": "reporter",
                "route to: finish": "finish",
                "route to: end": "finish"
            }
            
            # Look for routing directives
            for keyword, route in routing_keywords.items():
                if keyword in supervisor_content:
                    next_agent = route
                    logger.info(f"Found explicit routing directive: {keyword}")
                    break
                
            # If no explicit routing found, look for implicit intentions
            if not next_agent:
                # Check for finish signals
                if any(kw in supervisor_content for kw in ["finish", "done", "complete", "task completed"]):
                    next_agent = "finish"
                # Check for agent mentions
                elif "research" in supervisor_content:
                    next_agent = "researcher"
                elif "plan" in supervisor_content:
                    next_agent = "planner"
                elif "code" in supervisor_content:
                    next_agent = "coder"
                elif "browse" in supervisor_content:
                    next_agent = "browser"
                elif "report" in supervisor_content:
                    next_agent = "reporter"
                else:
                    # Default to coordinator if no clear direction
                    next_agent = "coordinator"
                
            # Before finishing, check if we've used required agents
            if next_agent == "finish":
                must_use_agents = ["RESEARCHER", "REPORTER"]
                missing_agents = [agent for agent in must_use_agents if agent not in state.get("contributing_agents", [])]
                
                if missing_agents:
                    logger.info(f"Missing required agent contributions: {missing_agents}. Routing to {missing_agents[0].lower()}")
                    return missing_agents[0].lower()
                else:
                    # Log performance metrics before finishing
                    if "execution_metrics" in state:
                        logger.info("Workflow performance metrics:")
                        for agent, time_taken in state["execution_metrics"].items():
                            logger.info(f"  {agent}: {time_taken:.2f}s")
                    
                    logger.info("All required agents have contributed. Finishing workflow.")
                    return "finish"
            
            # Return the determined next node
            logger.info(f"Routing to: {next_agent}")
            return next_agent
        
        except Exception as e:
            logger.error(f"Error in route_from_supervisor: {str(e)}")
            logger.error(traceback.format_exc())
            # Default to coordinator in case of errors
            return "coordinator"
    
    # Add conditional edges from coordinator
    builder.add_conditional_edges(
        "coordinator",
        lambda state: "supervisor",  # Always route to supervisor for centralized control
    )
    
    # Add conditional edges from planner
    builder.add_conditional_edges(
        "planner",
        lambda state: "supervisor",  # Always route back to supervisor for next step decision
    )
    
    # Add conditional edges from supervisor (central routing)
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor
    )
    
    # Add conditional edges from researcher
    builder.add_conditional_edges(
        "researcher",
        lambda state: "supervisor",  # Route back to supervisor for next step
    )
    
    # Add conditional edges from coder
    builder.add_conditional_edges(
        "coder",
        lambda state: "supervisor",  # Route back to supervisor for next step
    )
    
    # Add conditional edges from browser
    builder.add_conditional_edges(
        "browser",
        lambda state: "supervisor",  # Route back to supervisor for next step
    )
    
    # Add conditional edges from reporter
    builder.add_conditional_edges(
        "reporter",
        lambda state: "supervisor",  # Route back to supervisor for next step
    )
    
    # Compile the graph
    workflow = builder.compile()
    
    return workflow

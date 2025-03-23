import logging
import json
import traceback
import re
from typing import Dict, List, Any, Optional, Tuple, Union, cast

import json_repair
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, FunctionMessage
from langgraph.types import Command

from src.llms.llm import get_llm_by_type
from src.config import TEAM_MEMBERS
from src.config.agents import AGENT_LLM_MAP
from src.prompts.template import apply_prompt_template
from src.tools.search import tavily_tool, default_search_tool
from src.utils.json_utils import repair_json_output, parse_structured_output
from src.utils.message_utils import normalize_message, normalize_messages
from .types import State, Router

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

RESPONSE_FORMAT = "Response from {}:\n\n<response>\n{}\n</response>\n\n*Please execute the next step.*"

def extract_content(response: Any) -> str:
    """
    Extract string content from various response types.
    
    Args:
        response: The response from an LLM, which could be:
            - A string
            - An AIMessage object with content attribute
            - A dictionary with content key
            - Any other object that can be converted to string
            
    Returns:
        The extracted content as a string
    """
    if hasattr(response, "content"):
        # Handle AIMessage or similar objects
        return response.content
    elif isinstance(response, dict) and "content" in response:
        # Handle dictionary responses
        return response["content"]
    elif isinstance(response, str):
        # Already a string
        return response
    else:
        # Fallback: convert to string
        return str(response)

# Function to simplify consistent message creation
def create_agent_message(agent_name: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict:
    """
    Create a standardized agent message format compatible with LangChain and LM Studio.
    
    Args:
        agent_name: The name of the agent (e.g., "COORDINATOR", "RESEARCHER")
        content: The message content
        metadata: Optional metadata to attach to the message (e.g., routing information)
        
    Returns:
        A properly formatted message dictionary
    """
    # Truncate content if it's too long to conserve context
    max_content_length = 2000  # Characters, not tokens
    
    if len(content) > max_content_length:
        logger.warning(f"{agent_name} message exceeds length limit, truncating...")
        # Keep the beginning and end of the message, truncate the middle
        half_length = max_content_length // 2
        content = content[:half_length] + "\n...[content truncated]...\n" + content[-half_length:]
    
    # Format consistently for both LangChain and LM Studio
    return {
        "role": "assistant",
        "content": f"[{agent_name}]: {content}",
        "metadata": metadata or {}
    }

def research_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The researcher gathers information relevant to the problem."""
    logger.info("Running researcher")
    logger.debug(f"Researcher received {len(messages)} messages")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    researcher_prompt = apply_prompt_template("researcher", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [researcher_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["researcher"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Researcher raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with researcher's action
        state["actions"] = state.get("actions", "") + f"\n- Researcher gathered information relevant to the problem."
        
        # Create response message
        result = create_agent_message("RESEARCHER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in research_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("RESEARCHER", "Error occurred during research.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def code_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The coder writes and optimizes code to solve the problem."""
    logger.info("Running coder")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    coder_prompt = apply_prompt_template("coder", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [coder_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["coder"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Coder raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with coder's action
        state["actions"] = state.get("actions", "") + f"\n- Coder implemented code to solve the problem."
        
        # Create response message
        result = create_agent_message("CODER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in code_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("CODER", "Error occurred during code implementation.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def browser_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The browser interacts with web interfaces and retrieves information."""
    logger.info("Running browser")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    browser_prompt = apply_prompt_template("browser", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [browser_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["browser"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Browser raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with browser's action
        state["actions"] = state.get("actions", "") + f"\n- Browser retrieved web information relevant to the problem."
        
        # Create response message
        result = create_agent_message("BROWSER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in browser_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("BROWSER", "Error occurred during web browsing.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def coordinator_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The coordinator plans the overall approach and coordinates between team members."""
    logger.info("Running coordinator")
    logger.debug(f"Coordinator received messages: {messages}")
    logger.debug(f"Coordinator state: {state}")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    logger.debug(f"Coordinator state summary: {state_summary}")
    
    try:
        # Apply prompt template
        template_vars = {"state": state_summary}
        logger.debug(f"Sending to template: {template_vars}")
        coordinator_prompt = apply_prompt_template("coordinator", template_vars)
        
        # Normalize messages for consistency
        normalized_messages = normalize_messages(messages)
        
        # Combine messages
        all_messages = [coordinator_prompt, *normalized_messages]
        
        # Get response from LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Coordinator raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with coordinator's action
        state["actions"] = state.get("actions", "") + f"\n- Coordinator analyzed the request and coordinated the approach."
        
        # Create response message
        result = create_agent_message("COORDINATOR", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in coordinator_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("COORDINATOR", "Error occurred during coordination.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def planner_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The planner creates a detailed plan for solving the problem."""
    logger.info("Running planner")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    planner_prompt = apply_prompt_template("planner", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [planner_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["planner"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Planner raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with new plan
        state["plan"] = response
        state["actions"] = state.get("actions", "") + f"\n- Planner created a detailed plan for solving the problem."
        
        # Create response message
        result = create_agent_message("PLANNER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in planner_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("PLANNER", "Error occurred during planning.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def reporter_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The reporter creates a final report summarizing the findings and solution."""
    logger.info("Running reporter")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    reporter_prompt = apply_prompt_template("reporter", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [reporter_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Reporter raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with reporter's action
        state["actions"] = state.get("actions", "") + f"\n- Reporter summarized findings and created a final report."
        
        # Create response message with FINISH metadata
        result = create_agent_message("REPORTER", response, {"next": "FINISH"})
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in reporter_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message with FINISH metadata
        result = create_agent_message("REPORTER", "Error occurred during reporting.", {"next": "FINISH"})
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def supervisor_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The supervisor decides the next action based on the current state and messages."""
    logger.info("Running supervisor")
    logger.debug(f"Supervisor received messages: {len(messages)} messages")
    logger.debug(f"Supervisor state: {state.keys() if state else 'None'}")
    
    # Only keep the last 5 messages to prevent context overflow
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    logger.info(f"Using {len(recent_messages)} most recent messages to prevent context overflow")
    
    try:
        # Get the supervisor LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        
        # Apply prompt template with team members
        # Import team member configurations here to reduce import coupling
        from src.config import TEAM_MEMBER_CONFIGRATIONS
        team_members_desc = "\n".join([
            f"- {name}: {TEAM_MEMBER_CONFIGRATIONS[name]['desc']}" 
            for name in TEAM_MEMBERS
        ])
        template_vars = {"team_members": team_members_desc}
        logger.debug(f"Sending to template: {template_vars}")
        
        try:
            supervisor_prompt = apply_prompt_template("supervisor", template_vars)
        except Exception as template_error:
            logger.error(f"Error applying supervisor template: {template_error}")
            logger.error(traceback.format_exc())
            # Create a fallback prompt if template fails
            supervisor_prompt = {
                "role": "user", 
                "content": f"You are acting as the Supervisor agent. Based on the conversation so far, which team member should act next? Choose from: {', '.join(TEAM_MEMBERS)}, or FINISH."
            }
        
        # Normalize messages to ensure consistent format
        normalized_messages = normalize_messages(recent_messages)
        
        # Format the user context from normalized messages - just show the last request and responses
        user_context = "\n\n".join([
            f"[{msg.get('role', 'unknown').upper()}]: {msg.get('content', '')}" 
            for msg in normalized_messages
        ])
        
        # Create a shorter prompt with clear instructions
        final_prompt = "You are the workflow Supervisor. Review the recent conversation and decide which team member should act next.\n\n"
        final_prompt += f"Available team members: {', '.join(TEAM_MEMBERS)}, or FINISH if the task is complete.\n\n"
        final_prompt += "Recent conversation:\n" + user_context + "\n\n"
        
        # Add information about the current state to help with decision making
        state_summary = f"""
Current Status:
- Actions Taken: {len(state.get('actions', '').splitlines()) if isinstance(state.get('actions', ''), str) else 0}
- Has Plan: {"Yes" if state.get('plan', '') else "No"}
- Messages: {len(state.get('messages', []))}

Based on the above conversation and state, respond with EXACTLY ONE of:
{', '.join(TEAM_MEMBERS)}, or FINISH.

Your response format should be:
TEAM_MEMBER_NAME: Brief explanation
"""
        final_prompt += state_summary
        
        # Get response from LLM with a much smaller prompt
        raw_response = llm.invoke([{"role": "user", "content": final_prompt}])
        logger.debug(f"Supervisor raw response: {raw_response}")
        
        # Extract response content using our helper function
        response_content = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response_content}")
        
        # Parse the response to extract the next action
        next_action = "FINISH"  # Default if we can't parse
        
        # First try to extract team member name with regex
        role_match = re.search(r'^([A-Z]+):', response_content.strip(), re.MULTILINE)
        if role_match:
            candidate = role_match.group(1).upper()
            # Check if this is a valid team member or FINISH
            if candidate == "FINISH" or candidate in [tm.upper() for tm in TEAM_MEMBERS]:
                next_action = candidate
                logger.info(f"Found team member in response: {next_action}")
            
        # If no match with regex, look for team member names anywhere in the response
        if next_action == "FINISH" and not "finish" in response_content.lower():
            for team_member in TEAM_MEMBERS:
                # Look for team member name as a standalone word
                if re.search(r'\b' + team_member.lower() + r'\b', response_content.lower()):
                    next_action = team_member.upper()
                    logger.info(f"Found team member in response text: {next_action}")
                    break
                    
        # Always check for explicit FINISH indicators regardless of other matches
        finish_indicators = ["finish", "end", "complete", "done", "final"]
        if any(word in response_content.lower() for word in finish_indicators) and "next" not in response_content.lower():
            next_action = "FINISH"
            logger.info("Found FINISH indicator in response")
            
        logger.info(f"Supervisor routing to: {next_action}")
        
        # Make sure we're not in a loop by checking state history
        if "routing_history" not in state:
            state["routing_history"] = []
            
        # Add this routing decision to history
        state["routing_history"].append(next_action)
        
        # Check if we're in a loop (same route 3+ times in a row)
        if len(state["routing_history"]) >= 3:
            last_three = state["routing_history"][-3:]
            if all(r == last_three[0] for r in last_three) and next_action == last_three[0]:
                logger.warning(f"Detected routing loop to {next_action}, forcing FINISH")
                next_action = "FINISH"
        
        # Return formatted message with routing metadata
        result = create_agent_message("SUPERVISOR", 
            f"Next: {next_action}\nReasoning: {response_content}", 
            {"next": next_action}
        )
        
        # Update state messages to maintain consistency
        if "messages" in state and isinstance(state["messages"], list):
            # Only append our result if it's not already there
            if not any(
                isinstance(m, dict) and 
                m.get("role") == result["role"] and 
                m.get("content") == result["content"] 
                for m in state["messages"]
            ):
                state["messages"].append(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in supervisor_node: {e}")
        logger.error(f"Error traceback: {traceback.format_exc()}")
        logger.warning("Returning default FINISH response due to error")
        result = create_agent_message(
            "SUPERVISOR", 
            f"Error occurred during supervision: {str(e)}. Ending workflow.", 
            {"next": "FINISH"}
        )
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def writer_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """The writer creates content based on the gathered information."""
    logger.info("Running writer")
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    # Apply prompt template
    writer_prompt = apply_prompt_template("writer", {"state": state_summary})
    
    # Normalize messages for consistency
    normalized_messages = normalize_messages(messages)
    
    # Combine messages
    all_messages = [writer_prompt, *normalized_messages]
    
    # Get response from LLM
    try:
        llm = get_llm_by_type(AGENT_LLM_MAP["writer"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Writer raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Update state with writer's action
        state["actions"] = state.get("actions", "") + f"\n- Writer created content based on the gathered information."
        
        # Create response message
        result = create_agent_message("WRITER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in writer_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        result = create_agent_message("WRITER", "Error occurred during content creation.", None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

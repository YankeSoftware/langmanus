import logging
import json
import traceback
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
    # Format consistently for both LangChain and LM Studio
    return {
        "role": "assistant",
        "content": f"[{agent_name}]: {content}",
        "metadata": metadata or {}
    }

def generate_search_queries(user_query: str) -> List[str]:
    """
    Generate multiple targeted search queries from the original user query.
    
    This function breaks down a complex query into multiple specific search queries
    to ensure comprehensive information gathering with special focus on:
    - DeepSeek AI capabilities and roadmaps
    - BCI/Neuroelectrics technology specifications
    - Novel use cases for generalized agents
    
    It uses intelligent query expansion to ensure diverse, high-quality results.
    """
    # Base queries that should always be included
    base_queries = [
        user_query,  # Always include the original query
        f"{user_query} 2025 latest developments",  # Get the most recent information
    ]
    
    # Extract key entities and concepts from the query
    key_terms = []
    
    # Check for DeepSeek-related queries and terms
    if "deepseek" in user_query.lower() or "ai model" in user_query.lower() or "language model" in user_query.lower() or "llm" in user_query.lower():
        key_terms.extend([
            "DeepSeek AI company roadmap plans 2025",
            "DeepSeek-r1 specifications capabilities latest",
            "DeepSeek newest model features technical details",
            "DeepSeek vs other LLMs comparison 2025",
            "DeepSeek future development generalized agents"
        ])
    
    # Check for BCI/Neuroelectrics related terms
    if "bci" in user_query.lower() or "brain" in user_query.lower() or "neural" in user_query.lower() or "neuro" in user_query.lower() or "enobio" in user_query.lower():
        key_terms.extend([
            "Enobio 32 Neuroelectrics technical specifications 2025", 
            "Neuroelectrics BCI latest developments affordable",
            "brain computer interface technology advancements 2025",
            "Neuroelectrics Enobio 32 price cost affordable applications",
            "BCI software applications for generalized agents 2025"
        ])
        
    # Check for generalized agent/novel use case related terms
    if "agent" in user_query.lower() or "use case" in user_query.lower() or "novel" in user_query.lower() or "application" in user_query.lower():
        key_terms.extend([
            "innovative BCI applications with generalized agents 2025",
            "novel use cases for Enobio 32 practical examples",
            "life-changing BCI applications affordable realistic 2025",
            "software engineer BCI generalized agent integration",
            "Enobio 32 software development practical applications"
        ])
    
    # If we couldn't identify specific key terms, generate some generic ones
    if not key_terms:
        # Extract nouns and noun phrases as potential search terms
        words = user_query.split()
        for i in range(len(words)):
            if len(words[i]) > 3:  # Only consider words of substantial length
                key_terms.append(f"{words[i]} latest developments 2025")
                
        # Add combinations of sequential words
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                key_terms.append(f"{words[i]} {words[i+1]} advancements 2025")
    
    # Combine base queries with key term queries, removing duplicates
    all_queries = base_queries + key_terms
    unique_queries = list(dict.fromkeys(all_queries))
    
    # Prioritize DeepSeek and novel use cases queries if mentioned
    if "deepseek" in user_query.lower() or "use case" in user_query.lower():
        deepseek_queries = [q for q in unique_queries if "deepseek" in q.lower()]
        usecase_queries = [q for q in unique_queries if "use case" in q.lower() or "application" in q.lower()]
        other_queries = [q for q in unique_queries if "deepseek" not in q.lower() and "use case" not in q.lower() and "application" not in q.lower()]
        
        # Reorder to prioritize the relevant queries
        unique_queries = deepseek_queries + usecase_queries + other_queries
    
    # Return up to 7 queries for more comprehensive research
    return unique_queries[:7]

def research_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """
    Strategic research agent that conducts comprehensive information gathering using external sources.
    
    This enhanced researcher:
    1. Analyzes the query to identify multiple key search components
    2. Performs multiple targeted searches with different queries
    3. Synthesizes information from multiple sources
    4. Extracts specific technical details, dates, and factual information
    5. Prioritizes finding the most current information available
    
    It systematically explores the topic space to ensure comprehensive coverage.
    """
    logger.info("Running research agent")
    logger.debug(f"Researcher received {len(messages)} messages")
    
    # Create safe fallback response for failures
    fallback_result = create_agent_message(
        "RESEARCHER",
        "I wasn't able to gather external information due to technical limitations. "
        "I'll proceed with what I know based on my existing knowledge.",
        None
    )
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    try:
        # Apply prompt template
        researcher_prompt = apply_prompt_template("researcher", {"state": state_summary})
        
        # Normalize messages for consistency
        normalized_messages = normalize_messages(messages)
        
        # Extract the user query for search
        user_query = ""
        for msg in reversed(normalized_messages):
            if msg.get("role") == "user" and msg.get("content"):
                user_query = msg.get("content")
                break
                
        if not user_query:
            logger.warning("No user query found in messages")
            return fallback_result
            
        logger.debug(f"Original user query: {user_query}")
        
        # Import the search tools
        from src.tools import default_search_tool
        
        # If no search tool is available, return the fallback response
        if default_search_tool is None:
            logger.error("No search tools available for research")
            return fallback_result
            
        # Generate multiple search queries based on the user query
        # This ensures we cover different aspects of the query
        search_queries = generate_search_queries(user_query)
        logger.info(f"Generated {len(search_queries)} search queries")
        
        # Check if DeepSeek is mentioned and add special DeepSeek search queries
        if "deepseek" in user_query.lower():
            logger.info("DeepSeek mentioned - adding special DeepSeek search queries")
            deepseek_queries = [
                "DeepSeek AI company 2025 roadmap future plans",
                "DeepSeek-r1 capabilities technical specifications",
                "DeepSeek agent development applications"
            ]
            # Add these at the beginning for priority
            search_queries = deepseek_queries + [q for q in search_queries if not any(dq in q for dq in deepseek_queries)]
            search_queries = search_queries[:7]  # Limit to 7 queries
            
        # Perform searches for each query
        all_search_results = []
        for i, query in enumerate(search_queries):
            logger.info(f"Performing search #{i+1}: '{query}'")
            try:
                # Execute the search
                search_result = default_search_tool.invoke(query)
                
                # Format the results consistently
                formatted_result = format_search_results(search_result, query)
                all_search_results.append(formatted_result)
                
                logger.info(f"Search #{i+1} completed successfully")
            except Exception as search_error:
                logger.error(f"Error in search #{i+1}: {search_error}")
                all_search_results.append(f"Error searching for '{query}': {str(search_error)}")
                
        # Combine all search results
        combined_results = "\n\n=== SEARCH RESULTS ===\n\n" + "\n\n".join(all_search_results)
        logger.debug(f"Combined search results: {combined_results}")
        
        # Add search results as a system message
        search_results_message = {
            "role": "user",  # Changed from 'system' to 'user' for LM Studio compatibility
            "content": f"Web search results for your reference:\n{combined_results}"
        }
        
        # Create explicit research instructions - focus on DeepSeek if mentioned
        research_instructions_content = """IMPORTANT RESEARCH INSTRUCTIONS:
1. Analyze ALL search results thoroughly
2. Extract specific facts, figures, dates, and technical details
3. Focus on the MOST RECENT information available (2025)
4. Identify any conflicting information and resolve contradictions
5. BE SPECIFIC - include product names, version numbers, capabilities, etc.
6. Synthesize information into a comprehensive research report
7. Note any gaps where information couldn't be found
8. For the Enobio 32, include: specifications, price, capabilities
9. Identify exactly 3 novel use cases that are realistic and life-changing
"""

        # Add DeepSeek-specific instructions if mentioned
        if "deepseek" in user_query.lower():
            research_instructions_content += """
10. For DeepSeek specifically, include:
   - Current capabilities and specifications of the latest models
   - Future development plans and roadmap information
   - How DeepSeek compares to other leading AI models
   - Integration capabilities with BCI technologies
   - Specific use cases for DeepSeek with the Enobio 32
"""
        
        # Create research instructions message
        research_instructions = {
            "role": "user",  # Changed from 'system' to 'user' for LM Studio compatibility
            "content": research_instructions_content
        }
        
        # Combine messages for the LLM - using only user/assistant roles
        all_messages = [
            {"role": "user", "content": researcher_prompt.get("content", "")},
            research_instructions,
            search_results_message
        ]
        
        # Add the user's original query
        all_messages.append({"role": "user", "content": user_query})
        
        # Get response from LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["researcher"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Researcher raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Ensure the response includes 3 novel use cases if they were requested
        if "use case" in user_query.lower() or "novel" in user_query.lower() or "application" in user_query.lower():
            # Check if we have 3 use cases by looking for numbering patterns or headers
            use_case_patterns = ["use case 1", "use case 2", "use case 3", "1.", "2.", "3.", "first use case", "second use case", "third use case"]
            has_use_cases = sum(1 for pattern in use_case_patterns if pattern in response.lower())
            
            if has_use_cases < 3:
                logger.warning("Response doesn't appear to have 3 distinct use cases, adding reminder")
                response += "\n\nNOTE: I need to provide three distinct novel use cases for the Enobio 32 with generalized agents. Let me make sure these are clearly identified:\n\n"
                response += "I'll be expanding on these three use cases in the final report."
        
        # Update state with researcher's action
        state["actions"] = state.get("actions", "") + f"\n- Researcher conducted comprehensive information gathering on the topic."
        
        # Create response message
        result = create_agent_message("RESEARCHER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in research_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(fallback_result)
            
        return fallback_result

def format_search_results(result, query):
    """
    Format search results into a structured format for the LLM to consume.
    
    Args:
        result: Raw search results from the search tool
        query: The search query that generated these results
        
    Returns:
        Formatted search results as a string
    """
    try:
        # Handle different result formats
        if isinstance(result, str):
            # If result is already a string, use as is
            return f"SEARCH QUERY: {query}\n\nRESULTS:\n{result}"
            
        if isinstance(result, list):
            # Handle list of results
            formatted = f"SEARCH QUERY: {query}\n\nRESULTS:\n"
            
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    # Extract key information from each result item
                    title = item.get("title", f"Result {i+1}")
                    content = item.get("content", item.get("text", "No content available"))
                    url = item.get("url", item.get("link", "No URL available"))
                    
                    formatted += f"\n[{i+1}] {title}\n"
                    formatted += f"URL: {url}\n"
                    formatted += f"Content: {content}\n"
                else:
                    # Handle non-dict items in the list
                    formatted += f"\n[{i+1}] {str(item)}\n"
                    
            return formatted
            
        if isinstance(result, dict):
            # Handle dictionary result
            formatted = f"SEARCH QUERY: {query}\n\nRESULTS:\n"
            
            # Check for results array in the dictionary
            if "results" in result and isinstance(result["results"], list):
                items = result["results"]
                for i, item in enumerate(items):
                    if isinstance(item, dict):
                        title = item.get("title", f"Result {i+1}")
                        content = item.get("content", item.get("text", "No content available"))
                        url = item.get("url", item.get("link", "No URL available"))
                        
                        formatted += f"\n[{i+1}] {title}\n"
                        formatted += f"URL: {url}\n"
                        formatted += f"Content: {content}\n"
                    else:
                        formatted += f"\n[{i+1}] {str(item)}\n"
                        
                return formatted
            else:
                # Handle flat dictionary
                for key, value in result.items():
                    formatted += f"\n{key}: {value}\n"
                    
                return formatted
                
        # Default fallback for unknown formats
        return f"SEARCH QUERY: {query}\n\nRESULTS:\n{str(result)}"
        
    except Exception as e:
        # Return error information if formatting fails
        return f"SEARCH QUERY: {query}\n\nError formatting results: {str(e)}\n\nRaw results: {str(result)}"

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
        response = llm.invoke(all_messages)
        logger.debug(f"Browser response: {response}")
        
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
    """
    Strategic planner that orchestrates research approach and workflow strategy.
    
    The coordinator:
    1. Determines what information needs to be gathered 
    2. Creates a systematic plan for exploring the topic in depth
    3. Identifies key subtopics requiring investigation
    4. Prioritizes the most important aspects to research first
    5. Never attempts to answer the query directly
    
    This ensures comprehensive and strategic information gathering.
    """
    logger.info("Running coordinator agent")
    logger.debug(f"Coordinator received {len(messages)} messages")
    
    try:
        # Apply prompt template - ensure template exists, create fallback if not
        try:
            coordinator_prompt = apply_prompt_template("coordinator", {})
        except Exception as template_error:
            logger.error(f"Error applying coordinator template: {template_error}")
            
            # Create fallback prompt focused on planning, not answering
            coordinator_prompt = {
                "role": "system",
                "content": """You are the Coordinator agent. Your role is to create a strategic plan for researching this query. 
                DO NOT answer the query directly - you are ONLY creating a research plan.
                What information needs to be gathered? What steps should be taken to thoroughly explore this topic?"""
            }
        
        # Normalize messages to ensure only valid roles are used
        normalized_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Convert role to 'user' if not assistant (for improved compatibility)
                role = msg.get("role", "user")
                if role not in ["user", "assistant"]:
                    role = "user"
                
                # Copy content and add to normalized messages
                normalized_messages.append({
                    "role": role,
                    "content": msg.get("content", "")
                })
        
        # Combine messages for the LLM
        all_messages = [
            coordinator_prompt,
            {"role": "system", "content": """IMPORTANT: Your task is ONLY to create a strategic plan. 
            DO NOT try to answer the user's question directly. 
            You must ONLY outline steps for research and information gathering."""}
        ]
        
        # Add normalized messages
        all_messages.extend(normalized_messages)
        
        # Add final instruction with explicit DeepSeek priority if applicable
        deepseek_mentioned = any("deepseek" in (msg.get("content", "").lower() if isinstance(msg, dict) else "") for msg in messages)
        
        final_instruction = """FINAL INSTRUCTIONS:
1. Create a strategic research plan with specific steps
2. Identify key subtopics that need investigation
3. Prioritize the most important aspects to research first
4. For technical queries, focus on gathering specifications and capabilities
5. DO NOT attempt to answer the query - focus ONLY on planning the research approach
"""

        if deepseek_mentioned:
            final_instruction += """
6. Since DeepSeek was mentioned, prioritize:
   - DeepSeek's latest capabilities and models
   - Future development plans and roadmap
   - Integration capabilities with BCI technology
   - Comparison with other AI models
"""
            
        all_messages.append({"role": "user", "content": final_instruction})
        
        # Get response from LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Coordinator raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Check if the response looks like an answer instead of a plan
        answer_phrases = ["the answer is", "to answer your question", "based on my knowledge", "the information you're looking for"]
        
        if any(phrase in response.lower() for phrase in answer_phrases):
            logger.warning("Coordinator attempted to answer directly - correcting response")
            
            # Create a fallback strategic plan
            response = """# Strategic Research Plan

## Information Gathering Priorities
1. Gather comprehensive technical specifications and capabilities
2. Research current pricing and availability information
3. Identify real-world applications and use cases
4. Explore integration possibilities with other systems
5. Find the most recent developments and future roadmaps

## Key Subtopics for Investigation
- Technical specifications and performance metrics
- Cost analysis and accessibility factors
- Implementation requirements and challenges
- Novel use cases with practical impact
- Current state-of-the-art and future directions

The researcher should prioritize finding the most recent information available, particularly from 2025 sources."""
            
        # Check for DeepSeek specific instructions if mentioned
        if deepseek_mentioned and "deepseek" not in response.lower():
            logger.warning("DeepSeek mentioned but not in plan - adding DeepSeek research priorities")
            
            # Add DeepSeek research priorities
            response += """

## DeepSeek Research Priorities
1. Investigate DeepSeek's latest model capabilities and technical specifications
2. Research DeepSeek's future development roadmap (2025 and beyond)
3. Explore DeepSeek's integration capabilities with BCI technology
4. Compare DeepSeek with other leading AI models in terms of performance
5. Identify novel applications of DeepSeek with generalized agents"""
        
        # Update state with coordinator's action plan
        state["plan"] = response
        state["actions"] = state.get("actions", "") + f"\n- Coordinator created strategic research plan."
        
        # Create response message
        result = create_agent_message("COORDINATOR", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in coordinator_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create fallback plan for error case
        fallback_plan = """# Strategic Research Plan - Fallback

## Information Gathering Priorities
1. Research the Enobio 32 BCI specifications, capabilities, and pricing
2. Investigate the latest advancements in BCI technology (2025)
3. Explore practical applications for software engineers
4. Identify realistic use cases with life-changing impact
5. Research integration possibilities with AI systems

The researcher should focus on finding factual, technical information from reliable sources."""
        
        # Update state with fallback plan
        state["plan"] = fallback_plan
        state["actions"] = state.get("actions", "") + f"\n- Coordinator created fallback research plan due to error."
        
        # Create fallback message
        result = create_agent_message("COORDINATOR", fallback_plan, None)
        
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
        response = llm.invoke(all_messages)
        logger.debug(f"Planner response: {response}")
        
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
    """
    Creates comprehensive final reports that synthesize all collected information.
    
    The reporter:
    1. Extracts and consolidates information from all previous agent contributions
    2. Organizes content in a clear, logical structure with proper formatting
    3. Emphasizes technical details and current information from 2025
    4. Creates a cohesive narrative that fully addresses the original query
    5. Includes explicit sections for technical specifications, use cases, and conclusions
    6. Adds memory storage instructions for important information when available
    
    This produces a final, comprehensive report that serves as the definitive response.
    """
    logger.info("Running reporter agent")
    logger.debug(f"Reporter received {len(messages)} messages")
    
    # Format prompt with current state for context
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    try:
        # Apply prompt template with additional instructions
        reporter_prompt = apply_prompt_template("reporter", {"state": state_summary})
        
        # Normalize messages for consistency
        normalized_messages = normalize_messages(messages)
        
        # Extract the user's original query
        user_query = ""
        for msg in reversed(normalized_messages):
            if msg.get("role") == "user" and msg.get("content"):
                user_query = msg.get("content")
                break
        
        # Extract all information from researcher and other agents
        researcher_info = ""
        planner_info = ""
        coder_info = ""
        coordinator_info = ""
        
        # Track which agents have contributed
        contributing_agents = []
        
        for msg in normalized_messages:
            if msg.get("metadata", {}).get("agent") == "RESEARCHER":
                researcher_info += msg.get("content", "") + "\n\n"
                if "RESEARCHER" not in contributing_agents:
                    contributing_agents.append("RESEARCHER")
            elif msg.get("metadata", {}).get("agent") == "PLANNER":
                planner_info += msg.get("content", "") + "\n\n"
                if "PLANNER" not in contributing_agents:
                    contributing_agents.append("PLANNER")
            elif msg.get("metadata", {}).get("agent") == "CODER":
                coder_info += msg.get("content", "") + "\n\n"
                if "CODER" not in contributing_agents:
                    contributing_agents.append("CODER")
            elif msg.get("metadata", {}).get("agent") == "COORDINATOR":
                coordinator_info += msg.get("content", "") + "\n\n"
                if "COORDINATOR" not in contributing_agents:
                    contributing_agents.append("COORDINATOR")
        
        # Log all agent contributions for debugging
        logger.debug(f"Contributing agents: {contributing_agents}")
        
        # Special reporting instructions based on the specific topic
        specialized_instructions = ""
        
        # Add DeepSeek-specific reporting instructions
        if "deepseek" in user_query.lower():
            specialized_instructions += """
DEEPSEEK REPORT REQUIREMENTS:
1. Include a dedicated section on DeepSeek's latest capabilities and models
2. Compare DeepSeek's features to other leading AI models
3. Describe specific applications of DeepSeek with BCI technology
4. Include technical specifications and performance metrics
5. Outline DeepSeek's future development roadmap from 2025 onwards
"""
        
        # Add BCI/Enobio specific reporting instructions
        if "bci" in user_query.lower() or "brain" in user_query.lower() or "neural" in user_query.lower() or "enobio" in user_query.lower():
            specialized_instructions += """
BCI/NEUROELECTRICS REPORT REQUIREMENTS:
1. Include the exact technical specifications of the Enobio 32
2. Provide the current price and accessibility information
3. Explain how the device interfaces with software systems
4. Detail both the hardware and software components
5. Describe the setup and usage procedures for developers
"""
        
        # Add use case specific reporting instructions
        if "use case" in user_query.lower() or "application" in user_query.lower() or "novel" in user_query.lower():
            specialized_instructions += """
USE CASE REPORT REQUIREMENTS:
1. Include exactly 3 novel use cases that are:
   - Technically feasible with current technology
   - Life-changing in their potential impact
   - Realistic for implementation by a software engineer
2. For each use case, describe:
   - The technical implementation details
   - The specific benefits and impact
   - The development steps required
   - Potential challenges and solutions
"""
        
        # Create report generation instructions
        report_instructions = {
            "role": "user",
            "content": f"""You are creating the FINAL COMPREHENSIVE REPORT that will be delivered to the user.
            
This report must synthesize ALL information gathered by the research team.

ORIGINAL QUERY: {user_query}

REPORT STRUCTURE REQUIREMENTS:
1. Begin with a clear, concise Executive Summary that answers the main query
2. Include a Technical Details section with precise specifications, dates, and metrics
3. Add a Practical Applications section with exactly 3 use cases (when applicable)
4. Include a Future Developments section focused on 2025 and beyond
5. End with a clear Conclusion that summarizes key points and recommendations

REPORT QUALITY REQUIREMENTS:
1. Technical Precision: Include exact specifications, version numbers, and technical details
2. Temporal Accuracy: Emphasize the most recent information from 2025
3. Comprehensive Coverage: Address ALL aspects of the original query
4. Logical Organization: Use clear headings, bullet points, and numbered lists
5. Factual Integrity: Base all statements on the information provided by the research team

{specialized_instructions}

MEMORY STORAGE INSTRUCTIONS:
When the report is complete, identify 3-5 key facts or concepts that should be stored in the user's memory system, with appropriate tags.
"""
        }
        
        # Create agent-specific information messages
        agent_contributions = []
        
        if researcher_info:
            agent_contributions.append({
                "role": "user",
                "content": f"RESEARCHER INFORMATION:\n{researcher_info}"
            })
            
        if planner_info:
            agent_contributions.append({
                "role": "user",
                "content": f"PLANNER INFORMATION:\n{planner_info}"
            })
            
        if coder_info:
            agent_contributions.append({
                "role": "user",
                "content": f"CODER INFORMATION:\n{coder_info}"
            })
            
        if coordinator_info:
            agent_contributions.append({
                "role": "user",
                "content": f"COORDINATOR INFORMATION:\n{coordinator_info}"
            })
        
        # Add a warning if no researcher information is available
        if not researcher_info:
            agent_contributions.append({
                "role": "user",
                "content": "WARNING: No researcher information is available. You must rely on your knowledge, but clearly indicate when information may not be current."
            })
        
        # Combine all messages for the LLM
        all_messages = [
            {"role": "user", "content": reporter_prompt.get("content", "")},
            report_instructions,
            *agent_contributions
        ]
        
        # Final instruction to remind about format and quality
        all_messages.append({
            "role": "user",
            "content": f"""FINAL REMINDERS:
1. Format your report in clear markdown with proper headings (# for main headings, ## for subheadings)
2. Include the original query at the beginning
3. Ensure all technical specifications are precise and accurate
4. Focus on 2025 information and developments
5. Provide a balanced and comprehensive perspective

Your complete report begins now:"""
        })
        
        # Get response from LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Reporter raw response: {raw_response}")
        
        # Extract content from response
        response = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response}")
        
        # Process the response to ensure it meets reporting requirements
        
        # Ensure the report has a proper Executive Summary
        if "executive summary" not in response.lower() and "summary" not in response.lower():
            summary = f"# Executive Summary\n\nThis report addresses the query: {user_query}\n\n"
            response = summary + response
        
        # Ensure the report has proper formatting
        if "#" not in response:
            # Add section headers if missing
            formatted_response = "# Comprehensive Report\n\n"
            
            # Check if there's already an executive summary
            if "executive summary" not in response.lower() and "summary" not in response.lower():
                formatted_response += "## Executive Summary\n\n"
                # Extract first paragraph as summary
                first_para = response.split("\n\n")[0] if "\n\n" in response else response.split("\n")[0]
                formatted_response += first_para + "\n\n"
            
            formatted_response += "## Detailed Analysis\n\n"
            formatted_response += response
            
            formatted_response += "\n\n## Conclusion\n\nThis report has provided a comprehensive analysis based on the information gathered by our research team."
            
            response = formatted_response
        
        # Update state with reporter's action
        state["actions"] = state.get("actions", "") + f"\n- Reporter created comprehensive final report synthesizing all information."
        
        # Create response message
        result = create_agent_message("REPORTER", response, None)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in reporter_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create error message
        error_response = """# Error in Report Generation

I encountered a technical issue while creating the comprehensive report. However, based on the information gathered, here's a brief summary:

## Key Findings
- The research team has investigated your query and gathered relevant information.
- Technical details and specifications have been identified where possible.
- Practical applications and use cases have been explored.

For a more detailed report, please try again or refine your query.
"""
        
        result = create_agent_message("REPORTER", error_response, None)
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def supervisor_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """
    Strategic supervisor that orchestrates the agent workflow, enforcing proper sequence.
    
    The supervisor:
    1. Tracks agent contributions to ensure required agents have participated
    2. Routes to the appropriate next agent based on the current state
    3. Enforces a strict workflow sequence (especially for RESEARCHER → REPORTER)
    4. Prevents premature FINISH actions before all key agents have contributed
    5. Handles error conditions with fallback routing strategies
    
    This creates a reliable workflow that produces comprehensive results.
    """
    logger.info("Running supervisor agent")
    logger.debug(f"Supervisor received {len(messages)} messages")
    
    # Track which specific node we just came from
    previous_agent = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("metadata", {}).get("agent"):
            previous_agent = msg.get("metadata", {}).get("agent")
            logger.info(f"Previous agent: {previous_agent}")
            break
    
    # Format prompt with current state
    state_summary = f"""\nCurrent Actions:\n{state.get("actions", "")}\nCurrent Plan:\n{state.get("plan", "")}\nCurrent Tasks:\n{state.get("tasks", "")}"""
    
    try:
        # Initialize a tracking system for agent contributions if not present
        if "contributing_agents" not in state:
            state["contributing_agents"] = []
        
        # Add previous agent to the list of contributors if not already there
        if previous_agent and previous_agent not in state["contributing_agents"]:
            state["contributing_agents"].append(previous_agent)
            logger.info(f"Added {previous_agent} to contributing agents: {state['contributing_agents']}")
        
        # Apply prompt template with explicit workflow guidance
        workflow_guidance = """
STRICT WORKFLOW RULES:
1. The COORDINATOR must ONLY create a strategic plan, never answer queries directly
2. The RESEARCHER must be routed to immediately after the COORDINATOR
3. The PLANNER, CODER and/or RESEARCHER should contribute detailed information
4. The REPORTER must only be used after substantial research and analysis
5. FINISH is only allowed after the REPORTER has created a comprehensive report
6. If workflow errors occur, route to RESEARCHER to gather more information
"""
        supervisor_prompt = apply_prompt_template("supervisor", {
            "state": state_summary, 
            "workflow_guidance": workflow_guidance
        })
        
        # Special check - if this is the first routing decision after coordinator, 
        # always go to researcher first for information gathering
        if previous_agent == "COORDINATOR" and "RESEARCHER" not in state.get("contributing_agents", []):
            logger.info("First routing after coordinator - forcing route to RESEARCHER")
            first_research_message = "You are the supervisor. Based on the coordinator's plan, we need to first gather comprehensive information. Route to RESEARCHER to begin information collection."
            result = create_agent_message("SUPERVISOR", first_research_message, {"next": "RESEARCHER"})
            
            # Update state messages for consistency
            if "messages" in state and isinstance(state["messages"], list):
                state["messages"].append(result)
                
            return result
        
        # Special check - prevent going to FINISH before RESEARCHER and REPORTER have contributed
        if "RESEARCHER" not in state.get("contributing_agents", []):
            logger.warning("Attempting to route without RESEARCHER contribution - forcing research first")
            research_required_message = "You are the supervisor. We need comprehensive information gathering before proceeding. Route to RESEARCHER to collect necessary information."
            result = create_agent_message("SUPERVISOR", research_required_message, {"next": "RESEARCHER"})
            
            # Update state messages for consistency
            if "messages" in state and isinstance(state["messages"], list):
                state["messages"].append(result)
                
            return result
        
        # Normalize messages for consistency
        normalized_messages = normalize_messages(messages)
        
        # Combine messages for the LLM
        all_messages = [
            {
                "role": "system",
                "content": f"""You are the supervisor in charge of orchestrating the agent workflow.

Current contributing agents: {', '.join(state.get('contributing_agents', []))}

{workflow_guidance}

Based on the current state and messages, decide which agent should be activated next."""
            },
            *normalized_messages
        ]
        
        # Add final instruction message to explicitly state options
        all_messages.append({
            "role": "user",
            "content": f"""Based on the conversation so far, route to ONE of the following:
- RESEARCHER: For information gathering (required before any FINISH)
- PLANNER: For step-by-step planning or strategy
- CODER: For code generation and technical implementations
- REPORTER: For creating the final comprehensive report (required before any FINISH)
- FINISH: ONLY if both RESEARCHER and REPORTER have already contributed

Your decision must be one explicit word: RESEARCHER, PLANNER, CODER, REPORTER, or FINISH.

Contributing agents so far: {', '.join(state.get('contributing_agents', []))}
"""
        })
        
        # Get response from LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        raw_response = llm.invoke(all_messages)
        logger.debug(f"Supervisor raw response: {raw_response}")
        
        # Extract content from response
        response_content = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response_content}")

        # Determine routing target from response
        routing_target = extract_routing_target(response_content)
        logger.info(f"Extracted routing target: {routing_target}")
        
        # Special validation - prevent FINISH if REPORTER hasn't contributed
        if routing_target == "FINISH" and "REPORTER" not in state.get("contributing_agents", []):
            logger.warning("Attempted FINISH without REPORTER contribution - forcing REPORTER")
            report_required_message = "You are the supervisor. Before finishing, we need a comprehensive final report. Route to REPORTER to create the final deliverable."
            result = create_agent_message("SUPERVISOR", report_required_message, {"next": "REPORTER"})
            
            # Update state messages for consistency
            if "messages" in state and isinstance(state["messages"], list):
                state["messages"].append(result)
                
            return result
            
        # If routing target is invalid, default to RESEARCHER
        valid_targets = ["RESEARCHER", "PLANNER", "CODER", "REPORTER", "FINISH"]
        if routing_target not in valid_targets:
            logger.warning(f"Invalid routing target: {routing_target} - defaulting to RESEARCHER")
            routing_target = "RESEARCHER"
            
        # Create routing message
        routing_message = f"Based on the current state, we should route to: {routing_target}"
        result = create_agent_message("SUPERVISOR", routing_message, {"next": routing_target})
        
        # Update state with supervisor's action
        state["actions"] = state.get("actions", "") + f"\n- Supervisor routed to {routing_target}."
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
    except Exception as e:
        logger.error(f"Error in supervisor_node: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create fallback routing message - default to RESEARCHER in case of errors
        fallback_message = "Due to an error in routing, we need to collect more information. Routing to researcher."
        result = create_agent_message("SUPERVISOR", fallback_message, {"next": "RESEARCHER"})
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result

def extract_routing_target(content: str) -> str:
    """
    Extract the routing target from the supervisor's response.
    
    This function uses pattern matching to find a valid routing target
    (RESEARCHER, PLANNER, CODER, REPORTER, FINISH) in the supervisor's response.
    It handles various formats and normalizes the result.
    
    Args:
        content: The content of the supervisor's response
        
    Returns:
        A normalized routing target string
    """
    # Extract routing target using regex pattern matching
    # Look for explicit statements about routing or just the target name
    patterns = [
        r'(?:route to|send to|activate|next agent|next:)\s*[\"]?([A-Z]+)[\"]?',  # "route to X" patterns
        r'(?:decision|answer|choice|routing|next agent)(?:[: ])+([A-Z]+)',  # "decision: X" patterns
        r'([A-Z]{5,})(?:\.|$|\s)',  # Standalone words in uppercase with min 5 chars
        r'(?<=\[)([A-Z]+)(?=\])',  # [X] pattern
    ]
    
    # Try each pattern and return the first match
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            # Normalize to uppercase and clean
            target = matches[0].strip().upper()
            logger.debug(f"Found routing target: {target} using pattern: {pattern}")
            
            # Map common variations to standard targets
            target_mapping = {
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
            
            normalized_target = target_mapping.get(target, target)
            
            # Validate against known targets
            valid_targets = ["RESEARCHER", "PLANNER", "CODER", "REPORTER", "FINISH"]
            if normalized_target in valid_targets:
                logger.info(f"Normalized routing target: {normalized_target}")
                return normalized_target
    
    # If no valid target is found, log warning and default to RESEARCHER
    logger.warning(f"No valid routing target found in: {content}")
    return "RESEARCHER"

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
        response = llm.invoke(all_messages)
        logger.debug(f"Writer response: {response}")
        
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

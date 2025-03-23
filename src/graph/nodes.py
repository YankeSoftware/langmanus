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

def research_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """
    Strategic research agent that conducts targeted information gathering using external sources.
    
    This node implements a progressive research approach:
    1. First performs targeted query extraction and refinement
    2. Conducts multi-source search with fallbacks
    3. Analyzes and synthesizes information before reporting
    
    It's designed to handle research tasks with both breadth and depth.
    """
    logger.info("Running research agent")
    logger.debug(f"Researcher received {len(messages)} messages")
    
    # Create safe fallback response for failures
    fallback_result = create_agent_message(
        "RESEARCHER", 
        "I was unable to perform the requested research due to technical limitations. "
        "Here's what I know based on my training knowledge: [general information about the topic]"
    )
    
    try:
        # 1. PREPARATION PHASE
        # Get the LLM with appropriate context size
        llm = get_llm_by_type(AGENT_LLM_MAP["researcher"])
        
        # Normalize incoming messages
        normalized_messages = normalize_messages(messages)
        
        # Extract core query and generate search queries
        search_query = ""
        for msg in normalized_messages:
            if msg.get("role") == "user":
                search_query = msg.get("content", "")
                break
        
        # If no direct user query found, use the last message content
        if not search_query and normalized_messages:
            search_query = normalized_messages[-1].get("content", "")
        
        # Generate a more targeted search query using a separate prompt
        try:
            query_extractor_prompt = {
                "role": "user",
                "content": f"""Extract the key research questions from this query. Format your response as 2-3 specific search queries.
                
                ORIGINAL QUERY: {search_query}
                
                SEARCH QUERIES:
                1. """
            }
            
            # Get refined search queries
            query_response = llm.invoke([query_extractor_prompt])
            refined_queries = extract_content(query_response).strip().split("\n")
            logger.debug(f"Generated refined queries: {refined_queries}")
            
            # Clean up the queries (remove numbering and leading/trailing whitespace)
            cleaned_queries = []
            for q in refined_queries:
                q = q.strip()
                if q:
                    # Remove numbering if present (e.g., "1. ", "2. ")
                    if q[0].isdigit() and len(q) > 2 and q[1:3] in [". ", ") "]:
                        q = q[3:].strip()
                    cleaned_queries.append(q)
            
            if cleaned_queries:
                # Use the refined queries instead of raw query
                logger.info(f"Using {len(cleaned_queries)} refined search queries")
            else:
                # Fallback to original query if refinement failed
                cleaned_queries = [search_query]
                
        except Exception as e:
            logger.warning(f"Failed to refine search queries: {e}")
            cleaned_queries = [search_query]
        
        # 2. SEARCH PHASE - Multi-tool search with fallbacks
        search_results = []
        
        # Get search tool from state or try to initialize one
        search_tool = None
        if "search_tool" in state and state["search_tool"] is not None:
            logger.debug("Using search tool from state")
            search_tool = state["search_tool"]
        else:
            # Try to fetch search tools with fallback mechanism
            logger.warning("No search tool in state, attempting to initialize one")
            try:
                # Try to import the get_best_search_tool function
                try:
                    from src.tools.search import get_best_search_tool
                    search_tool = get_best_search_tool()
                except (ImportError, AttributeError):
                    # Direct fallback to available tools
                    from src.tools.search import brave_tool, tavily_tool, default_search_tool
                    search_tool = default_search_tool
            except Exception as search_error:
                logger.error(f"Failed to initialize search tools: {search_error}")
        
        # Execute searches with the tool
        if search_tool:
            logger.info(f"Starting multi-query search process with {len(cleaned_queries)} queries")
            
            for i, query in enumerate(cleaned_queries):
                try:
                    logger.info(f"Executing search {i+1}/{len(cleaned_queries)}: {query}")
                    result = search_tool.invoke(query)
                    
                    if result and isinstance(result, str):
                        search_results.append(f"Results for '{query}':\n{result}")
                        logger.info(f"Search {i+1} successful")
                    else:
                        logger.warning(f"Search {i+1} returned empty or invalid result")
                except Exception as e:
                    logger.error(f"Search {i+1} failed: {e}")
                    
                    # Try fallback to Tavily if available and not already used
                    try:
                        from src.tools.search import tavily_tool
                        if tavily_tool and search_tool != tavily_tool:
                            logger.info(f"Trying fallback search with Tavily for query {i+1}")
                            fallback_result = tavily_tool.invoke(query)
                            if fallback_result:
                                search_results.append(f"Results for '{query}' (fallback):\n{fallback_result}")
                                logger.info(f"Fallback search {i+1} successful")
                    except Exception as fallback_error:
                        logger.error(f"Fallback search also failed: {fallback_error}")
        else:
            logger.warning("No search tool available - research will be limited to model knowledge")
        
        # 3. MEMORY INTEGRATION - Check memory for relevant information
        memory_results = []
        try:
            from src.tools.memory import query_memory
            memory_response = query_memory(search_query)
            if memory_response:
                logger.info("Found relevant information in memory")
                memory_results = [f"Information from memory: {memory_response}"]
        except Exception as memory_error:
            logger.warning(f"Error querying memory: {memory_error}")
        
        # 4. SYNTHESIS PHASE
        # Prepare research summary with all collected information
        all_results = memory_results + search_results
        
        # Add a note if no external information was found
        if not all_results:
            all_results = ["Note: External search was unavailable or returned no results. " +
                          "This response is based on the model's training knowledge."]
        
        # Get researcher prompt template
        researcher_prompt = apply_prompt_template("researcher", {"state": state})
        
        # Combine all components into a comprehensive research package
        synthesis_prompt = [
            researcher_prompt,
            *normalized_messages,
            {"role": "user", "content": 
             f"""I've gathered the following information based on your request:
             
             SEARCH RESULTS:
             {chr(10).join(all_results)}
             
             Please analyze this information and provide:
             1. A comprehensive summary of findings
             2. Key insights relevant to the original query
             3. Any limitations or gaps in the research
             
             Structure your response as a research report."""}
        ]
        
        # Generate the final research analysis
        raw_response = llm.invoke(synthesis_prompt)
        logger.debug(f"Research analysis raw response: {raw_response}")
        
        # Extract and format response
        response_content = extract_content(raw_response)
        logger.debug(f"Extracted research content: {response_content}")
        
        # Create a researcher message with the analysis
        result = create_agent_message("RESEARCHER", response_content)
        
        # Update state tracking
        state["actions"] = state.get("actions", "") + f"\n- Researcher conducted multi-source research on {len(cleaned_queries)} search queries."
        
        # Update state messages to maintain consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error in research_node: {e}")
        logger.error(traceback.format_exc())
        
        # Update state messages even in error case
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(fallback_result)
            
        return fallback_result

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
    Strategic coordinator that develops comprehensive execution plans and manages task dependencies.
    
    This enhanced coordinator:
    1. Performs in-depth task analysis and decomposition
    2. Establishes clear agent responsibilities and sequencing
    3. Creates structured execution plans with measurable outcomes
    4. Provides high-level coordination across the agent team
    
    It serves as the primary planner and decision-maker for the entire workflow.
    """
    logger.info("Running coordinator")
    logger.debug(f"Coordinator received messages: {messages}")
    logger.debug(f"Coordinator state: {state}")
    
    try:
        # Get the appropriate LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["coordinator"])
        
        # Normalize messages for consistent processing
        normalized_messages = normalize_messages(messages)
        
        # Extract user query for analysis (assuming first user message is the primary query)
        user_query = ""
        for msg in normalized_messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
                
        if not user_query and normalized_messages:
            # Fallback to last message if no user message found
            user_query = normalized_messages[-1].get("content", "")
            
        # Get team capabilities to inform planning
        from src.config import TEAM_MEMBER_CONFIGRATIONS
        team_capabilities = "\n".join([
            f"- {name}: {TEAM_MEMBER_CONFIGRATIONS[name]['desc']}" 
            for name in TEAM_MEMBERS
        ])
            
        # Generate a comprehensive workflow plan with 2-stage approach
        
        # Stage 1: Analyze the query and identify necessary steps
        analysis_prompt = {
            "role": "user",
            "content": f"""Analyze the following user query to identify required tasks, dependencies, and agents:

USER QUERY: {user_query}

Available Team Members:
{team_capabilities}

Please analyze this request and provide:
1. A detailed breakdown of the core problem and objectives
2. The key information or data needed to address the request
3. A list of 3-5 concrete tasks needed to fulfill this request
4. The dependencies between these tasks (which must be completed before others)
5. The most suitable team member to handle each task

Provide your analysis in this structured format:
PROBLEM: [breakdown of the core problem]
OBJECTIVES: [clear definition of success criteria]
INFORMATION NEEDED: [data or context required]
TASKS:
- [task 1]: [responsible agent]
- [task 2]: [responsible agent]
...
DEPENDENCIES:
- [task 2] depends on [task 1]
...
"""
        }
        
        # Get the analytical breakdown
        analysis_response = llm.invoke([analysis_prompt])
        analysis_content = extract_content(analysis_response)
        logger.debug(f"Query analysis: {analysis_content}")
        
        # Stage 2: Develop a concrete plan based on the analysis
        planning_prompt = {
            "role": "user",
            "content": f"""Based on the following analysis, create a comprehensive execution plan:

QUERY ANALYSIS:
{analysis_content}

Develop a step-by-step execution plan that:
1. Defines clear actions for each team member
2. Establishes measurable outcomes for each step
3. Provides specific instructions for information gathering, processing, and reporting
4. Ensures all user requirements will be satisfied

Your execution plan should be clear, thorough, and actionable.
"""
        }
        
        # Generate the execution plan
        plan_response = llm.invoke([planning_prompt])
        plan_content = extract_content(plan_response)
        logger.debug(f"Execution plan: {plan_content}")
        
        # Store the plan in state for future reference
        state["plan"] = plan_content
        
        # Extract specific tasks from the plan for tracking
        task_extraction_prompt = {
            "role": "user",
            "content": f"""Extract the specific tasks from this execution plan:

EXECUTION PLAN:
{plan_content}

List each concrete task along with:
1. The responsible agent
2. The expected outcome
3. Any dependencies (which tasks must be completed first)

Format your response as a numbered list of tasks ONLY.
"""
        }
        
        # Get the task list
        tasks_response = llm.invoke([task_extraction_prompt])
        tasks_content = extract_content(tasks_response)
        logger.debug(f"Extracted tasks: {tasks_content}")
        
        # Store tasks in state
        state["tasks"] = tasks_content
        
        # Initialize or update task tracking structure
        if "task_status" not in state:
            state["task_status"] = {
                "pending": [],
                "completed": [],
                "dependencies": {}
            }
            
        # Update pending tasks from the new task list
        tasks = [task.strip() for task in tasks_content.split("\n") if task.strip()]
        state["task_status"]["pending"] = [
            task for task in tasks 
            if task not in state["task_status"]["completed"]
        ]
        
        # Create a user-facing report summarizing the plan
        coordinator_prompt = apply_prompt_template("coordinator", {"state": state})
        
        # Generate final coordinator response
        summary_prompt = {
            "role": "user",
            "content": f"""Based on the user's request and our analysis, provide a clear summary of how we'll approach this task.

USER REQUEST: {user_query}

OUR ANALYSIS: 
{analysis_content}

OUR PLAN:
{plan_content}

Provide a concise, user-friendly response that:
1. Acknowledges their request
2. Outlines our approach at a high level (avoid technical details)
3. Sets clear expectations about what information we'll provide
4. Asks any clarifying questions if needed

Your response should be conversational and helpful.
"""
        }
        
        raw_response = llm.invoke([summary_prompt])
        logger.debug(f"Coordinator raw response: {raw_response}")
        
        # Extract and format content
        response_content = extract_content(raw_response)
        logger.debug(f"Extracted coordinator content: {response_content}")
        
        # Update state tracking
        state["actions"] = state.get("actions", "") + f"\n- Coordinator developed comprehensive plan and task assignments."
        
        # Create response message
        result = create_agent_message("COORDINATOR", response_content)
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error in coordinator_node: {e}")
        logger.error(traceback.format_exc())
        
        # Create error message
        result = create_agent_message("COORDINATOR", 
            "I apologize, but I encountered an issue while planning our approach. " + 
            "Let me try a simpler coordination strategy. How can I help with your request?", 
            None
        )
        
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
    Strategic reporter that synthesizes research and creates comprehensive reports with data-driven insights.
    
    This enhanced reporter:
    1. Performs in-depth analysis of all collected data
    2. Identifies key patterns, trends, and insights
    3. Creates structured, evidence-based reports with citations
    4. Provides visual representations of data when appropriate
    5. Includes actionable recommendations based on findings
    
    It serves as the final integration point for all agent outputs.
    """
    logger.info("Running reporter")
    logger.debug(f"Reporter received {len(messages)} messages")
    
    try:
        # Get the appropriate LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["reporter"])
        
        # Normalize messages for consistent processing
        normalized_messages = normalize_messages(messages)
        
        # Extract user query to understand reporting requirements
        user_query = ""
        for msg in normalized_messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
                
        if not user_query and normalized_messages:
            # Fallback to last message if no user message found
            user_query = normalized_messages[-1].get("content", "")
        
        # Extract all responses from previous agents, particularly researcher and coder
        agent_findings = {
            "researcher": [],
            "coder": [],
            "browser": [],
            "coordinator": []
        }
        
        # Collect information from each agent
        for msg in normalized_messages:
            content = msg.get("content", "")
            for agent in agent_findings.keys():
                # Check if this message is from this agent
                agent_upper = agent.upper()
                if f"[{agent_upper}]" in content:
                    agent_findings[agent].append(content)
        
        # Prepare a summary of findings from each agent
        findings_summary = ""
        for agent, findings in agent_findings.items():
            if findings:
                findings_summary += f"\n## {agent.upper()} FINDINGS\n"
                findings_summary += "\n".join(findings[-1:])  # Most recent finding
                findings_summary += "\n\n"
                
        # Check if we have a plan in the state
        plan = state.get("plan", "")
        if plan:
            findings_summary += f"\n## EXECUTION PLAN\n{plan}\n\n"
            
        # Check if we have a tasks list
        tasks = state.get("tasks", "")
        if tasks:
            findings_summary += f"\n## TASK PROGRESS\n{tasks}\n\n"
            
        # First stage: Analyze all findings
        analysis_prompt = {
            "role": "user",
            "content": f"""Analyze all findings from the agent team to identify key insights, patterns, and recommendations:

USER QUERY: {user_query}

COLLECTED FINDINGS:
{findings_summary}

Perform a comprehensive analysis that:
1. Identifies the most important and relevant information
2. Highlights connections between different pieces of information
3. Detects patterns, trends, or gaps in the collected data
4. Determines the most valuable insights for the user
5. Formulates evidence-based recommendations

Format your analysis as:
KEY INSIGHTS: [list the most important discoveries]
PATTERNS: [identify any trends or relationships]
GAPS: [note any missing information or limitations]
RECOMMENDATIONS: [provide actionable suggestions based on findings]
"""
        }
        
        # Get the analysis
        analysis_response = llm.invoke([analysis_prompt])
        analysis_content = extract_content(analysis_response)
        logger.debug(f"Report analysis: {analysis_content}")
        
        # Second stage: Create comprehensive report
        report_prompt = {
            "role": "user",
            "content": f"""Create a comprehensive report based on the following analysis:

USER QUERY: {user_query}

ANALYSIS:
{analysis_content}

RAW FINDINGS:
{findings_summary}

Create a detailed, well-structured report that:
1. Starts with an executive summary of key findings
2. Organizes information in logical sections with clear headings
3. Includes concrete evidence supporting each conclusion
4. Provides proper attribution for information sources
5. Ends with actionable recommendations and next steps

Your report should be professional, evidence-based, and directly address the user's query.
"""
        }
        
        # Get the report
        report_response = llm.invoke([report_prompt])
        report_content = extract_content(report_response)
        logger.debug(f"Generated report: {report_content}")
        
        # Apply final template formatting from prompt template
        reporter_prompt = apply_prompt_template("reporter", {"state": state})
        
        # Format final response with template
        final_response = llm.invoke([
            reporter_prompt,
            {"role": "user", "content": f"""
Here is the comprehensive report I've prepared:

{report_content}

Please review this report and ensure it:
1. Directly addresses the user's original query: "{user_query}"
2. Presents information in a clear, logical structure
3. Provides actionable insights and recommendations
4. Is professional and polished in tone

If needed, make any final adjustments to improve clarity, structure, or presentation.
"""}
        ])
        
        # Extract and format the final response
        final_content = extract_content(final_response)
        logger.debug(f"Final formatted report: {final_content}")
        
        # Update state tracking
        state["actions"] = state.get("actions", "") + f"\n- Reporter synthesized findings and created comprehensive final report."
        
        # Create response message with FINISH metadata
        result = create_agent_message("REPORTER", final_content, {"next": "FINISH"})
        
        # Update state messages for consistency
        if "messages" in state and isinstance(state["messages"], list):
            state["messages"].append(result)
            
        return result
        
    except Exception as e:
        logger.error(f"Error in reporter_node: {e}")
        logger.error(traceback.format_exc())
        
        # Create a simplified report based on whatever information we have
        try:
            # Get user query from messages
            user_query = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break
                    
            # Create a basic report
            fallback_content = f"""
# Report on: {user_query}

I apologize, but I encountered an issue while creating a comprehensive report. 
Here's a simplified summary based on the information gathered so far:

## Key Findings
- The team investigated your query about {user_query}
- We gathered information from multiple sources
- Our analysis indicates [basic information about the topic]

## Recommendations
- Consider [basic recommendation 1]
- [basic recommendation 2]
- For more detailed information, specific follow-up queries may be helpful

Thank you for your understanding.
"""
            # Create response message with FINISH metadata
            result = create_agent_message("REPORTER", fallback_content, {"next": "FINISH"})
            
            # Update state messages even in error case
            if "messages" in state and isinstance(state["messages"], list):
                state["messages"].append(result)
                
            return result
            
        except Exception as fallback_error:
            logger.error(f"Fallback report also failed: {fallback_error}")
            
            # Create most basic error message with FINISH metadata
            result = create_agent_message(
                "REPORTER", 
                "I apologize, but I encountered an issue while creating your report. " +
                "Please try again with more specific questions or contact support for assistance.",
                {"next": "FINISH"}
            )
            
            # Update state messages even in this error case
            if "messages" in state and isinstance(state["messages"], list):
                state["messages"].append(result)
                
            return result

def supervisor_node(messages: List, state: Dict, agent_config: Dict) -> Dict:
    """
    Strategic supervisor that orchestrates the agent workflow using intelligent coordination.
    
    This enhanced supervisor uses:
    1. Explicit task decomposition and dependency tracking
    2. Adaptive routing based on task progress and dependencies
    3. Contextual memory of previous decisions for coherent orchestration
    
    It manages the team of specialized agents to ensure efficient collaboration.
    """
    logger.info("Running supervisor")
    logger.debug(f"Supervisor received messages: {len(messages)} messages")
    logger.debug(f"Supervisor state: {state.keys() if state else 'None'}")
    
    # Only keep the last 5 messages to prevent context overflow
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    logger.info(f"Using {len(recent_messages)} most recent messages to prevent context overflow")
    
    try:
        # Get the supervisor LLM
        llm = get_llm_by_type(AGENT_LLM_MAP["supervisor"])
        
        # Get team member descriptions
        from src.config import TEAM_MEMBER_CONFIGRATIONS
        team_members_desc = "\n".join([
            f"- {name}: {TEAM_MEMBER_CONFIGRATIONS[name]['desc']}" 
            for name in TEAM_MEMBERS
        ])
        
        # Track supervision history if it doesn't exist yet
        if "supervision_history" not in state:
            state["supervision_history"] = []
            
        # Track task status and dependencies
        if "task_status" not in state:
            state["task_status"] = {
                "pending": [],
                "completed": [],
                "dependencies": {}
            }
            
        # Create task progress summary
        task_progress = ""
        if state["task_status"]["completed"]:
            task_progress += "Completed Tasks:\n"
            task_progress += "\n".join([f"- {task}" for task in state["task_status"]["completed"]])
            task_progress += "\n\n"
            
        if state["task_status"]["pending"]:
            task_progress += "Pending Tasks:\n"
            task_progress += "\n".join([f"- {task}" for task in state["task_status"]["pending"]])
            task_progress += "\n\n"
            
        # Add previous routing decisions with reasoning to provide context
        routing_history = ""
        if state["supervision_history"]:
            routing_history = "Previous Coordinator Decisions:\n"
            # Only include the last 3 decisions to avoid context bloat
            for i, (agent, reason) in enumerate(state["supervision_history"][-3:]):
                routing_history += f"{i+1}. Routed to {agent} because {reason}\n"
            routing_history += "\n"
            
        # Normalize messages for consistent formatting
        normalized_messages = normalize_messages(recent_messages)
        
        # Analyze the current state to determine progress and next steps
        # First, attempt to understand the user query and what's been done so far
        analysis_prompt = {
            "role": "user",
            "content": f"""Analyze the current state of the workflow to determine strategic next steps:

Current Team Members:
{team_members_desc}

{task_progress}
{routing_history}

Recent Messages:
{"".join([f'\n[{msg.get("role", "").upper()}]: {msg.get("content", "")}' for msg in normalized_messages])}

Based on this information, what are the current goals of this workflow, what progress has been made,
and what are the next 1-2 strategic steps needed?

Respond in this exact format:
GOALS: [list the primary goals of this workflow]
PROGRESS: [summarize what has been accomplished so far]
NEXT STEPS: [1-2 specific next steps that should be taken]
"""
        }
        
        # Get strategic analysis
        analysis_response = llm.invoke([analysis_prompt])
        analysis_content = extract_content(analysis_response)
        logger.debug(f"Strategic analysis: {analysis_content}")
        
        # Now use the analysis to determine the next agent
        routing_prompt = {
            "role": "user",
            "content": f"""As the workflow supervisor, you need to decide which agent should act next.

Current Team Members:
{team_members_desc}

Strategic Analysis:
{analysis_content}

{task_progress}
{routing_history}

Decide which agent should act next. Respond with ONLY the agent name followed by a brief explanation.
Valid agents: {', '.join(TEAM_MEMBERS)}, or FINISH if the workflow is complete.

Your response format: [AGENT_NAME]: [brief explanation why this agent should act next]
"""
        }
        
        # Get routing decision
        raw_response = llm.invoke([routing_prompt])
        logger.debug(f"Supervisor raw response: {raw_response}")
        
        # Extract response content
        response_content = extract_content(raw_response)
        logger.debug(f"Extracted response content: {response_content}")
        
        # Parse the response to extract the next action and reasoning
        next_action = "FINISH"  # Default if we can't parse
        reasoning = response_content
        
        # Use regex to extract agent name and reasoning
        import re
        agent_match = re.match(r'^([A-Z]+):\s*(.*)', response_content.strip(), re.DOTALL)
        if agent_match:
            agent_name = agent_match.group(1).strip().upper()
            reasoning = agent_match.group(2).strip()
            
            # Check if this is a valid team member or FINISH
            valid_agents = [tm.upper() for tm in TEAM_MEMBERS] + ["FINISH"]
            if agent_name in valid_agents:
                next_action = agent_name
                logger.info(f"Supervisor decided on {next_action}: {reasoning[:50]}...")
        
        # If no clear decision, look for team member names or FINISH in the response
        if next_action == "FINISH" and "finish" not in response_content.lower():
            # Check for team member names in the text
            for team_member in TEAM_MEMBERS:
                if re.search(r'\b' + team_member.lower() + r'\b', response_content.lower()):
                    next_action = team_member.upper()
                    logger.info(f"Found team member in response: {next_action}")
                    break
        
        # Check if we're in a loop
        if "routing_history" not in state:
            state["routing_history"] = []
            
        # Add this routing decision to history
        state["routing_history"].append(next_action)
        if len(state["routing_history"]) >= 3:
            last_three = state["routing_history"][-3:]
            if all(r == last_three[0] for r in last_three):
                logger.warning(f"Detected routing loop to {next_action}, forcing progression")
                # Find a team member we haven't used recently
                for member in TEAM_MEMBERS:
                    if member.upper() != next_action and member.upper() not in last_three:
                        next_action = member.upper()
                        reasoning += f"\n[LOOP DETECTED: Forcing progression to {next_action}]"
                        break
                        
                # If all team members are in the loop, finish the workflow
                if all(r == last_three[0] for r in last_three):
                    next_action = "FINISH"
                    reasoning += "\n[LOOP DETECTED: All agents in loop, finishing workflow]"
        
        # Store the decision in supervision history 
        if next_action != "FINISH":
            state["supervision_history"].append((next_action, reasoning[:100]))
            
        logger.info(f"Supervisor routing to: {next_action}")
        
        # Update state tracking
        state["actions"] = state.get("actions", "") + f"\n- Supervisor strategically routed to {next_action}."
        
        # Return formatted message with routing metadata
        result = create_agent_message("SUPERVISOR", 
            f"Next: {next_action}\nReasoning: {reasoning}", 
            {"next": next_action}
        )
        
        # Update state messages to maintain consistency
        if "messages" in state and isinstance(state["messages"], list):
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

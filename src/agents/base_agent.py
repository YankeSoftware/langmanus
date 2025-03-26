"""
Base agent class that serves as a foundation for all specialized agents.

This module provides a consistent interface for all agents in the system,
ensuring proper prompt loading from templates and interactions with LM Studio
or other model backends.
"""

import logging
import traceback
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.config.models import get_model_for_agent
from src.tools.tool_manager import get_tools_for_agent
from src.prompts.template import apply_prompt_template

logger = logging.getLogger(__name__)

# Define custom exception hierarchy for better error handling
class AgentError(Exception):
    """Base exception for all agent-related errors."""
    pass

class AgentInitializationError(AgentError):
    """Exception raised when an agent fails to initialize."""
    pass

class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    pass

class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out."""
    pass

class ModelAPIError(AgentError):
    """Exception raised when there's an error with the LLM API."""
    pass

class BaseAgent:
    """Base agent class that provides common functionality for all agents."""
    
    def __init__(self, agent_type: str, max_iterations: int = 5, api_config: Dict[str, Any] = None):
        """
        Initialize the base agent with tools and appropriate model.
        
        Args:
            agent_type: Type of agent (RESEARCHER, PLANNER, etc.)
            max_iterations: Maximum number of iterations for agent execution
            api_config: Optional configuration for API endpoints
        """
        self.agent_type = agent_type
        self.api_config = api_config or {}
        
        try:
            # Get the appropriate model for this agent type
            self.model = get_model_for_agent(agent_type, api_config)
            
            # Get the agent-specific tools
            self.tools = get_tools_for_agent(agent_type)
            
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
                max_iterations=max_iterations,
                early_stopping_method="force",
            )
        except Exception as e:
            error_msg = f"Failed to initialize {agent_type} agent: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise AgentInitializationError(error_msg) from e
    
    def extract_query(self, messages: List[Dict[str, Any]]) -> str:
        """
        Extract the query from the messages.
        
        Args:
            messages: List of messages in the conversation
            
        Returns:
            The extracted query
        """
        query = ""
        
        # Try to find the most recent user message
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                query = message.get("content", "")
                break
            elif isinstance(message, dict) and message.get("type") == "human":
                query = message.get("content", "")
                break
            elif isinstance(message, HumanMessage):
                query = message.content
                break
        
        # If no user message found, look for any message with content
        if not query:
            for message in reversed(messages):
                if isinstance(message, dict) and message.get("content"):
                    query = message.get("content", "")
                    break
                elif hasattr(message, "content") and message.content:
                    query = message.content
                    break
        
        return query
    
    def prepare_agent_input(self, messages: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the input for the agent.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            
        Returns:
            Agent input with properly formatted messages
        """
        try:
            # Get the query
            query = self.extract_query(messages)
            
            # Apply the prompt template from the prompts folder
            system_message = apply_prompt_template(self.agent_type.lower(), state)
            
            # Create input for the agent
            return {
                "messages": [system_message, HumanMessage(content=query)],
            }
        except Exception as e:
            error_msg = f"Failed to prepare input for {self.agent_type} agent: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise AgentExecutionError(error_msg) from e
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and perform agent-specific tasks.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Agent's response
        """
        try:
            # Extract the query to log
            query = self.extract_query(messages)
            logger.info(f"{self.agent_type} processing query: {query[:50]}...")
            
            # Prepare agent input
            agent_input = self.prepare_agent_input(messages, state)
            
            # Run the agent with timeout handling
            try:
                # Track execution in state for monitoring
                if "agent_executions" not in state:
                    state["agent_executions"] = []
                state["agent_executions"].append({"agent": self.agent_type, "timestamp": "now", "status": "started"})
                
                result = self.agent_executor.invoke(agent_input)
                
                # Update execution status
                state["agent_executions"][-1]["status"] = "completed"
                
                # Extract the result
                output = result.get("output", "")
                
                # Return the result with metadata
                return {
                    "content": output,
                    "metadata": {
                        "agent": self.agent_type,
                        "next": self.determine_next_step(output, state),
                        "execution_successful": True
                    }
                }
            except Exception as e:
                # Update execution status
                if "agent_executions" in state and state["agent_executions"]:
                    state["agent_executions"][-1]["status"] = "failed"
                    state["agent_executions"][-1]["error"] = str(e)
                
                # Check for specific error types
                if "timeout" in str(e).lower():
                    logger.error(f"Timeout in {self.agent_type} agent execution: {str(e)}")
                    raise AgentTimeoutError(f"Execution timed out for {self.agent_type} agent") from e
                elif "api" in str(e).lower() or "connection" in str(e).lower() or "model" in str(e).lower():
                    logger.error(f"API error in {self.agent_type} agent execution: {str(e)}")
                    raise ModelAPIError(f"Model API error for {self.agent_type} agent: {str(e)}") from e
                else:
                    logger.error(f"Error in {self.agent_type} agent execution: {str(e)}")
                    raise AgentExecutionError(f"Execution error for {self.agent_type} agent: {str(e)}") from e
                
        except (AgentTimeoutError, ModelAPIError, AgentExecutionError) as e:
            # Handle known error types with specific recovery strategies
            logger.error(f"{type(e).__name__} in {self.agent_type} agent: {str(e)}")
            
            # Implement graceful degradation
            fallback_message = f"I encountered an issue while processing your request as {self.agent_type}. "
            
            if isinstance(e, AgentTimeoutError):
                fallback_message += "The operation took too long to complete. I'll provide a simplified response."
            elif isinstance(e, ModelAPIError):
                fallback_message += "There was an issue connecting to the AI service. I'll try to answer with limited capabilities."
            else:
                fallback_message += f"Error details: {str(e)}"
            
            return {
                "content": fallback_message,
                "metadata": {
                    "agent": self.agent_type, 
                    "next": "COORDINATOR",
                    "execution_successful": False,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            }
        except Exception as e:
            # Handle unexpected errors
            logger.exception(f"Unexpected error in {self.agent_type} agent")
            return {
                "content": f"I encountered an unexpected error while processing your request: {str(e)}",
                "metadata": {
                    "agent": self.agent_type, 
                    "next": "COORDINATOR",
                    "execution_successful": False,
                    "error_type": "UnexpectedError",
                    "error_message": str(e)
                }
            }
    
    def determine_next_step(self, output: str, state: Dict[str, Any]) -> str:
        """
        Determine the next agent to handle the task based on output.
        
        Args:
            output: The output from the agent
            state: Current state of the conversation
            
        Returns:
            The next agent to route to
        """
        # This is a generic implementation that should be overridden by subclasses
        return "SUPERVISOR" 
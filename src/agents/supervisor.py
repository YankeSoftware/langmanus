import logging
import json
import time
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage
from src.config import TEAM_MEMBERS
from src.agents.base_agent import BaseAgent
from src.utils.message_utils import normalize_messages, normalize_message

logger = logging.getLogger(__name__)

class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that oversees the multi-agent workflow, making decisions
    about which agent should handle each part of the task and when to 
    terminate the process.
    
    The supervisor is responsible for:
    1. Tracking progress of the overall task
    2. Detecting potential deadlocks or circular handoffs
    3. Ensuring all agent contributions are properly integrated
    4. Deciding when the task is complete
    """
    
    def __init__(self, api_config: Dict[str, Any] = None):
        """
        Initialize the supervisor agent with appropriate model.
        
        Args:
            api_config: Optional configuration for API endpoints
        """
        super().__init__("SUPERVISOR", max_iterations=3, api_config=api_config)
        
        # Track previous agent assignments for deadlock detection
        self.previous_assignments = []
        self.deadlock_threshold = 3
        
        # Track start time for timeout detection
        self.start_time = time.time()
        self.timeout_threshold = 300  # 5 minutes, configurable via env var
        
        # Standard format for interagent communication
        self.message_format = {
            "type": "agent_message",
            "sender": "SUPERVISOR",
            "timestamp": 0,
            "content": "",
            "metadata": {}
        }
    
    def detect_deadlock(self, next_agent: str, state: Dict[str, Any]) -> bool:
        """
        Detect potential deadlocks in the agent workflow.
        
        Args:
            next_agent: The next agent to be assigned
            state: Current workflow state
            
        Returns:
            True if deadlock is detected, False otherwise
        """
        # Track agent assignments
        self.previous_assignments.append(next_agent)
        
        # Only check if we have enough assignments
        if len(self.previous_assignments) < self.deadlock_threshold * 2:
            return False
            
        # Check last N assignments to see if we're in a loop
        last_n = self.previous_assignments[-self.deadlock_threshold:]
        assignments_before = self.previous_assignments[-(self.deadlock_threshold*2):-self.deadlock_threshold]
        
        # If the same sequence repeats, we may have a deadlock
        if last_n == assignments_before:
            logger.warning(f"Potential deadlock detected: {last_n} repeated")
            return True
            
        # Check if we've been ping-ponging between two agents
        if len(set(last_n)) == 2 and len(last_n) >= 4:
            logger.warning(f"Potential ping-pong deadlock between {set(last_n)}")
            return True
            
        return False
    
    def track_progress(self, state: Dict[str, Any]) -> float:
        """
        Track progress of the overall task.
        
        Args:
            state: Current workflow state
            
        Returns:
            Progress percentage (0-100)
        """
        # Initialize progress tracking if not present
        if "progress" not in state:
            state["progress"] = {
                "total_steps": 5,  # Estimated based on typical workflow
                "completed_steps": 0,
                "current_step": "initialization"
            }
            
        # Update progress based on contributing agents
        contributing_agents = state.get("contributing_agents", [])
        
        # Critical path for most tasks
        critical_path = ["RESEARCHER", "PLANNER", "CODER", "REPORTER"]
        completed_critical = sum(1 for agent in critical_path if agent in contributing_agents)
        
        # Calculate percentage
        progress_percentage = min(100, (completed_critical / len(critical_path)) * 100)
        
        # Update state
        state["progress"]["completed_steps"] = completed_critical
        state["progress"]["percentage"] = progress_percentage
        
        return progress_percentage
    
    def check_timeout(self) -> bool:
        """Check if the workflow has exceeded the timeout threshold."""
        elapsed_time = time.time() - self.start_time
        return elapsed_time > self.timeout_threshold
    
    def format_agent_message(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format message in standardized format for inter-agent communication."""
        message = self.message_format.copy()
        message["content"] = content
        message["timestamp"] = time.time()
        message["metadata"] = metadata
        return message
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and oversee the workflow.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Supervisor's decision on next steps
        """
        try:
            # Track progress
            progress = self.track_progress(state)
            logger.info(f"Current task progress: {progress:.1f}%")
            
            # Check for timeout
            if self.check_timeout():
                logger.warning("Workflow timeout detected")
                return {
                    "content": "I've detected that this task is taking longer than expected. Let me summarize the current progress and provide a final response based on what we've learned so far.",
                    "metadata": {
                        "agent": "SUPERVISOR",
                        "next": "REPORTER",
                        "reason": "timeout"
                    }
                }
            
            # Call the parent implementation
            result = super().process_messages(messages, state, metadata)
            
            # Extract the next agent
            next_agent = result.get("metadata", {}).get("next", "COORDINATOR")
            
            # Check for deadlock
            if self.detect_deadlock(next_agent, state):
                logger.warning(f"Deadlock detected, redirecting to REPORTER instead of {next_agent}")
                
                # Break the deadlock by redirecting to reporter
                deadlock_message = "I've detected a potential loop in our workflow where agents are repeating the same steps without making progress. Let me summarize what we've learned so far and provide a response based on the current information."
                
                # Record deadlock in state for analysis
                state["workflow_issues"] = state.get("workflow_issues", []) + [
                    {
                        "type": "deadlock",
                        "detected_at": time.time(),
                        "pattern": self.previous_assignments[-self.deadlock_threshold:],
                        "action": "redirected_to_reporter"
                    }
                ]
                
                return {
                    "content": deadlock_message,
                    "metadata": {
                        "agent": "SUPERVISOR",
                        "next": "REPORTER",
                        "reason": "deadlock"
                    }
                }
                
            # Format the result in standardized inter-agent format
            metadata = result.get("metadata", {})
            metadata["progress"] = progress
            
            # Store data in standardized format for other agents
            standardized_message = self.format_agent_message(result["content"], metadata)
            if "inter_agent_messages" not in state:
                state["inter_agent_messages"] = []
            state["inter_agent_messages"].append(standardized_message)
            
            return result
            
        except Exception as e:
            logger.exception(f"Error in supervisor agent: {e}")
            return {
                "content": f"I encountered an error while supervising the workflow: {str(e)}. Let me redirect to a specialist who can help.",
                "metadata": {
                    "agent": "SUPERVISOR", 
                    "next": "COORDINATOR",
                    "error": str(e)
                }
            }
            
    def determine_next_step(self, output: str, state: Dict[str, Any]) -> str:
        """
        Determine which agent should handle the next step based on the current state.
        
        Args:
            output: The output from the supervisor
            state: Current state of the conversation
            
        Returns:
            Name of the next agent to route to
        """
        # Set up fallback paths in case routing fails
        fallback_paths = {
            "research_needed": "RESEARCHER",
            "planning_needed": "PLANNER",
            "implementation_needed": "CODER",
            "web_interaction_needed": "BROWSER",
            "coordination_needed": "COORDINATOR",
            "completion_needed": "REPORTER"
        }
        
        # Check for specific directives in the output
        output_lower = output.lower()
        
        # Handle explicit keywords for each agent
        if "research" in output_lower or "search" in output_lower or "information" in output_lower:
            return "RESEARCHER"
        elif "plan" in output_lower or "strategy" in output_lower or "approach" in output_lower:
            return "PLANNER"
        elif "code" in output_lower or "implement" in output_lower or "program" in output_lower:
            return "CODER"
        elif "browser" in output_lower or "web" in output_lower or "navigate" in output_lower:
            return "BROWSER"
        elif "coordinate" in output_lower or "manage" in output_lower or "assign" in output_lower:
            return "COORDINATOR"
        elif "report" in output_lower or "summary" in output_lower or "conclusion" in output_lower or "finish" in output_lower:
            return "REPORTER"
            
        # If no clear directive, base decision on current progress
        progress = state.get("progress", {}).get("percentage", 0)
        contributing_agents = state.get("contributing_agents", [])
        
        if progress < 25 and "RESEARCHER" not in contributing_agents:
            return "RESEARCHER"
        elif progress < 50 and "PLANNER" not in contributing_agents:
            return "PLANNER"
        elif progress < 75 and "CODER" not in contributing_agents:
            return "CODER"
        elif progress >= 75 or "finish" in output_lower or "complete" in output_lower:
            return "REPORTER"
        
        # Default fallback
        return "COORDINATOR"

def supervisor_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supervisor node function for the graph.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the supervisor's analysis and next agent recommendation
    """
    api_config = state.get("api_config", {})
    supervisor = SupervisorAgent(api_config)
    return supervisor.process_messages(messages, state, metadata) 
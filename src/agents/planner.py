import logging
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.models import get_model_for_agent

logger = logging.getLogger(__name__)

class PlannerAgent:
    """
    Planner agent that develops strategic plans and approaches for complex tasks.
    This agent excels at breaking down problems, identifying dependencies, and creating roadmaps.
    """
    
    def __init__(self, api_config: Dict[str, Any] = None):
        """
        Initialize the planner agent with models and templates.
        
        Args:
            api_config: Optional configuration for API endpoints
        """
        # Use DeepSeek for planning if available (requires complex reasoning)
        self.model = get_model_for_agent("PLANNER", api_config)
        
        # Create the system prompt template
        self.system_prompt = """You are an expert Planner Agent who excels at developing comprehensive, strategic approaches to complex problems.

Your role is to analyze tasks, break them down into manageable components, identify dependencies, and create detailed execution plans.

Follow these guidelines:

1. ANALYSIS APPROACH:
   - Begin by thoroughly understanding the problem space and requirements
   - Identify key constraints, resources, and success criteria
   - Consider multiple potential approaches before selecting the optimal strategy
   - Look for hidden assumptions and potential obstacles
   - Ensure your plan covers all aspects of the task

2. PLAN STRUCTURE:
   - Start with a clear, concise summary of the overall approach
   - Break down the task into logical phases with clear objectives
   - Identify dependencies between different components
   - Include estimated timelines or sequence information when relevant
   - Specify required resources, tools, or knowledge for each step
   - Highlight critical decision points and alternative paths
   - Include validation steps to verify successful execution

3. PLANNING PRINCIPLES:
   - Prioritize steps based on impact, dependencies, and effort
   - Balance thoroughness with efficiency
   - Include contingency plans for high-risk steps
   - Consider scalability and future maintenance when relevant
   - Leverage existing tools, frameworks, and patterns when appropriate
   - Design for robustness and error handling
   - Document key design decisions and their rationales

4. CONTENT GUIDELINES:
   - Be specific and actionable - avoid vague directives
   - Use consistent terminology throughout the plan
   - Provide concrete examples for complex or abstract concepts
   - Include reference information or links to relevant resources
   - Structure information using clear headings, numbering, and formatting
   - Use diagrams or visual representations when they add clarity (described textually)
   - Balance high-level strategy with tactical details

5. CONTEXT AWARENESS:
   - Consider the expertise level of those who will execute the plan
   - Acknowledge when additional research may be needed and specify what to investigate
   - Identify when specialized expertise might be required
   - Note any assumptions made in developing the plan
   - Highlight areas of uncertainty that may need further clarification

Your primary goal is to create plans that are:
1. Comprehensive - addressing all aspects of the task
2. Actionable - providing clear guidance for implementation
3. Efficient - minimizing wasted effort and resources
4. Robust - accounting for potential obstacles and edge cases
5. Adaptable - allowing for adjustment as conditions change

Always conclude your plan with:
1. A "Critical Success Factors" section highlighting key elements essential for success
2. A "Next Steps" section with immediate actions to begin implementation
3. A clear recommendation for which agent should continue the task
"""
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and generate a strategic plan.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Dict containing the plan and routing metadata
        """
        try:
            # Extract the query and context
            query = ""
            context = []
            
            # First pass to get the original query
            for message in messages:
                if isinstance(message, dict) and message.get("type") == "human":
                    query = message.get("content", "")
                    break
                elif isinstance(message, HumanMessage):
                    query = message.content
                    break
            
            # Second pass to collect relevant context from other agents
            for message in messages:
                if isinstance(message, dict) and isinstance(message.get("metadata"), dict):
                    agent = message.get("metadata", {}).get("agent")
                    content = message.get("content", "")
                    
                    if agent and agent != "PLANNER" and content:
                        # Truncate long messages to keep context manageable
                        if len(content) > 2000:
                            truncated_content = content[:2000] + "... [content truncated]"
                            context.append(f"--- {agent} CONTRIBUTION ---\n{truncated_content}\n")
                        else:
                            context.append(f"--- {agent} CONTRIBUTION ---\n{content}\n")
            
            # Check if we have a valid query
            if not query:
                return {
                    "content": "I couldn't find a clear task to plan for. Please provide a specific question or task.",
                    "metadata": {"agent": "PLANNER", "next": "COORDINATOR"}
                }
            
            # Combine the context
            combined_context = "\n\n".join(context) if context else "No additional context available."
            
            # Initialize planning state if not already present
            if "planning" not in state:
                state["planning"] = {
                    "task_analyzed": False,
                    "approach_selected": False,
                    "plan_created": False,
                    "approach_type": None,
                }
            
            # Create the prompt for generating the plan
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", f"""Task: {query}

Additional Context from Other Agents:
{combined_context}

Please analyze this task and develop a comprehensive strategic plan for approaching it.
Break down the problem into logical components, identify dependencies, and create a detailed roadmap.
Consider both the big picture strategy and the tactical details needed for successful execution.
Be sure to include critical success factors and immediate next steps.
""")
            ])
            
            # Generate the plan
            response = self.model.invoke(prompt)
            
            # Extract the content
            plan_content = response.content if hasattr(response, 'content') else str(response)
            
            # Update planning state
            state["planning"]["task_analyzed"] = True
            state["planning"]["plan_created"] = True
            
            # Determine next agent based on the plan content
            next_agent = self._determine_next_agent(plan_content, state)
            
            # Return the plan with metadata
            return {
                "content": plan_content,
                "metadata": {"agent": "PLANNER", "next": next_agent}
            }
            
        except Exception as e:
            logger.error(f"Error in planner agent: {str(e)}")
            return {
                "content": f"I encountered an error while developing a plan: {str(e)}",
                "metadata": {"agent": "PLANNER", "next": "COORDINATOR"}
            }
    
    def _determine_next_agent(self, plan_content: str, state: Dict[str, Any]) -> str:
        """
        Determine which agent should handle the task next based on plan content.
        
        Args:
            plan_content: The content of the generated plan
            state: Current state with planning information
            
        Returns:
            The next agent to route to
        """
        lower_content = plan_content.lower()
        
        # Look for explicit recommendations in the content
        if "next agent: researcher" in lower_content or "route to researcher" in lower_content:
            return "RESEARCHER"
        elif "next agent: coder" in lower_content or "route to coder" in lower_content:
            return "CODER"
        elif "next agent: browser" in lower_content or "route to browser" in lower_content:
            return "BROWSER"
        elif "next agent: reporter" in lower_content or "route to reporter" in lower_content:
            return "REPORTER"
        
        # If no explicit recommendation, infer from content
        if "code" in lower_content or "implement" in lower_content or "programming" in lower_content:
            return "CODER"
        elif "research" in lower_content or "investigate" in lower_content or "gather information" in lower_content:
            return "RESEARCHER"
        elif "browser" in lower_content or "navigate" in lower_content or "web interface" in lower_content:
            return "BROWSER"
        
        # Default to coordinator for further routing
        return "SUPERVISOR"

# Create the planner agent instance
planner_agent = PlannerAgent()

def planner_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner node function for the workflow graph.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the planner's analysis and plan
    """
    api_config = state.get("api_config", {})
    planner = PlannerAgent(api_config)
    return planner.process_messages(messages, state, metadata) 
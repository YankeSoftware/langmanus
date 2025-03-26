import logging
import re
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.models import get_model_for_agent
from src.tools.tool_manager import get_tools_for_agent

logger = logging.getLogger(__name__)

class CoderAgent:
    """
    Coder agent that implements technical solutions based on plans and requirements.
    This agent excels at writing code, designing software architecture, and technical implementation.
    """
    
    def __init__(self):
        """Initialize the coder agent with models and templates."""
        # Get DeepSeek model for coding (requires complex reasoning)
        self.model = get_model_for_agent("CODER")
        
        # Get relevant tools for the coder
        self.tools = get_tools_for_agent("CODER")
        
        # Create the system prompt template
        self.system_prompt = """You are an expert Coder Agent who excels at implementing technical solutions through code.

Your role is to transform plans, requirements, and specifications into working code and technical implementations.

Follow these guidelines:

1. IMPLEMENTATION APPROACH:
   - Begin by understanding the requirements and technical constraints
   - Break down complex problems into manageable components
   - Consider efficiency, maintainability, and scalability
   - Follow standard coding practices and design patterns
   - Write code that is clean, documented, and well-structured

2. CODE QUALITY:
   - Write code that is readable and self-documenting
   - Include appropriate comments for complex logic
   - Follow consistent naming conventions and formatting
   - Handle errors and edge cases gracefully
   - Create modular, reusable components
   - Ensure proper input validation and security practices

3. TECHNICAL CONSIDERATIONS:
   - Select appropriate languages, frameworks, and libraries for the task
   - Consider performance implications of implementation choices
   - Design for maintainability and future extensions
   - Follow relevant architectural patterns (MVC, microservices, etc.)
   - Consider compatibility across platforms when relevant
   - Optimize critical paths while maintaining readability

4. RESPONSE FORMAT:
   - Begin with a brief overview of your implementation approach
   - Present code blocks with proper syntax highlighting using triple backticks
   - Break down complex solutions into multiple files/modules as needed
   - Include instructions for running/deploying the code
   - Explicitly mention any dependencies or setup requirements
   - Provide examples of how to use the implemented code
   - Include testing approaches or test cases when relevant

5. CONTEXT AWARENESS:
   - Adapt your coding style to the existing codebase if one exists
   - Consider the skill level of the intended code users
   - Note any assumptions made while implementing
   - Highlight areas where additional input might be needed
   - Consider practical implementation constraints

Your primary goal is to create technical solutions that are:
1. Functional - correctly implementing the specified requirements
2. Maintainable - following good software engineering practices
3. Efficient - making appropriate use of resources
4. Secure - adhering to relevant security best practices
5. Well-documented - enabling others to understand and use the code

Always conclude your implementation with:
1. A brief summary of what you've implemented
2. Any limitations or considerations for future enhancements
3. A clear recommendation for what should happen next
"""
    
    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and implement a technical solution.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Dict containing the code implementation and routing metadata
        """
        try:
            # Extract the query and context
            query = ""
            context = []
            plan_content = ""
            research_content = ""
            
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
                    
                    if agent and content:
                        # Store plan and research content separately for focused context
                        if agent == "PLANNER":
                            plan_content = content
                        elif agent == "RESEARCHER":
                            research_content += content + "\n\n"
                        
                        # Truncate long messages to keep context manageable
                        if len(content) > 2000:
                            truncated_content = content[:2000] + "... [content truncated]"
                            context.append(f"--- {agent} CONTRIBUTION ---\n{truncated_content}\n")
                        else:
                            context.append(f"--- {agent} CONTRIBUTION ---\n{content}\n")
            
            # Check if we have a valid query
            if not query:
                return {
                    "content": "I couldn't find a clear task to implement. Please provide specific requirements or a technical task to code.",
                    "metadata": {"agent": "CODER", "next": "COORDINATOR"}
                }
            
            # Initialize coding state if not already present
            if "coding" not in state:
                state["coding"] = {
                    "requirements_analyzed": False,
                    "architecture_designed": False,
                    "implementation_created": False,
                    "technologies": [],
                }
            
            # Determine if we need to extract technical requirements first
            if not state["coding"]["requirements_analyzed"] and not plan_content:
                # We don't have a plan, so we need to extract requirements from the query and research
                requirements_extraction_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are an expert at extracting technical requirements from user requests and research materials.
Your task is to identify the core technical needs, constraints, and specifications for a software implementation.
Present your findings in a structured format with clear, specific requirements.
Focus on functional requirements, technical constraints, and expected behaviors."""),
                    ("human", f"""Original Query: {query}

Research Information:
{research_content}

Please extract the technical requirements for implementing a solution to this query.
Identify specific features, behaviors, technologies, and constraints that should guide the implementation.
Structure your response as a clear requirements document.
""")
                ])
                
                # Extract requirements
                requirements_response = self.model.invoke(requirements_extraction_prompt)
                
                # Store the extracted requirements
                plan_content = "EXTRACTED TECHNICAL REQUIREMENTS:\n\n" + (requirements_response.content if hasattr(requirements_response, 'content') else str(requirements_response))
                state["coding"]["requirements_analyzed"] = True
                
                # Add to context
                context.append(f"--- EXTRACTED REQUIREMENTS ---\n{plan_content}\n")
            
            # Combine the context, prioritizing plan content
            combined_context = ""
            if plan_content:
                combined_context += f"IMPLEMENTATION PLAN/REQUIREMENTS:\n{plan_content}\n\n"
            combined_context += "\n\n".join(context) if context else "No additional context available."
            
            # Create the prompt for generating the implementation
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", f"""Task: {query}

Context and Requirements:
{combined_context}

Please implement a technical solution that addresses the requirements and specifications provided.
Write clean, well-documented code with appropriate structure and organization.
Include instructions for running/deploying the code and any necessary setup steps.
""")
            ])
            
            # Generate the implementation
            response = self.model.invoke(prompt)
            
            # Extract the content
            implementation_content = response.content if hasattr(response, 'content') else str(response)
            
            # Update coding state
            state["coding"]["implementation_created"] = True
            
            # Extract technologies used using a simple regex approach
            tech_pattern = r'(?i)(python|javascript|typescript|java|c\+\+|ruby|go|rust|react|angular|vue|node\.js|django|flask|spring|express|mongodb|postgresql|mysql|tensorflow|pytorch)'
            tech_matches = re.findall(tech_pattern, implementation_content)
            if tech_matches:
                state["coding"]["technologies"] = list(set(tech_match.lower() for tech_match in tech_matches))
            
            # Determine next agent based on the implementation content
            next_agent = self._determine_next_agent(implementation_content)
            
            # Return the implementation with metadata
            return {
                "content": implementation_content,
                "metadata": {"agent": "CODER", "next": next_agent}
            }
            
        except Exception as e:
            logger.error(f"Error in coder agent: {str(e)}")
            return {
                "content": f"I encountered an error while implementing the solution: {str(e)}",
                "metadata": {"agent": "CODER", "next": "SUPERVISOR"}
            }
    
    def _determine_next_agent(self, implementation_content: str) -> str:
        """
        Determine which agent should handle the task next based on implementation content.
        
        Args:
            implementation_content: The content of the generated implementation
            
        Returns:
            The next agent to route to
        """
        lower_content = implementation_content.lower()
        
        # Look for explicit recommendations in the content
        if "next agent: researcher" in lower_content or "route to researcher" in lower_content:
            return "RESEARCHER"
        elif "next agent: planner" in lower_content or "route to planner" in lower_content:
            return "PLANNER"
        elif "next agent: browser" in lower_content or "route to browser" in lower_content:
            return "BROWSER"
        elif "next agent: reporter" in lower_content or "route to reporter" in lower_content:
            return "REPORTER"
        
        # If implementation needs testing/execution, route to browser
        if "test in browser" in lower_content or "run in browser" in lower_content:
            return "BROWSER"
        
        # If more research is specifically requested
        if "need more research" in lower_content or "additional information needed" in lower_content:
            return "RESEARCHER"
        
        # Default to reporter to summarize the solution
        return "REPORTER"

# Create the coder agent instance
coder_agent = CoderAgent()

def code_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming messages in the coder node.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the code implementation and routing metadata
    """
    return coder_agent.process_messages(messages, state, metadata) 
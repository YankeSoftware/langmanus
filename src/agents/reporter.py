import logging
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from src.config.models import get_model_for_agent

logger = logging.getLogger(__name__)

class ReporterAgent:
    """
    Reporter agent that synthesizes findings from other agents and produces 
    well-structured final reports with actionable conclusions.
    """
    
    def __init__(self):
        """Initialize the reporter with models and templates."""
        # Get the appropriate model (local or DeepSeek)
        self.model = get_model_for_agent("REPORTER")
        
        # Create the system prompt template
        self.system_prompt = """You are an expert Reporter Agent who excels at summarizing complex information.

Your job is to synthesize all the information provided by other agents into a clear, coherent, and comprehensive report.

Follow these guidelines:

1. SYNTHESIS APPROACH:
   - Extract key insights from each agent's contributions
   - Identify patterns, themes, and connections across different agent outputs
   - Resolve contradictions when possible, or acknowledge them when they persist
   - Prioritize information by relevance and reliability
   - Ensure the final report is coherent and flows logically

2. STRUCTURE:
   - Begin with an Executive Summary (1-2 paragraphs)
   - Organize the main body by themes/topics, not by agent
   - Include sections for: Background, Key Findings, Analysis, and Recommendations
   - Use clear headings and subheadings
   - End with a clear Conclusion section
   - Add an "Information Gaps" section for unresolved questions

3. CONTENT GUIDELINES:
   - Maintain factual accuracy - don't introduce new speculative information
   - Attribute information to sources when relevant
   - Distinguish between well-established facts and uncertain information
   - Use objective, clear language
   - Include relevant metrics, statistics, and data when available
   - Remove redundancies while preserving important nuances
   - Highlight actionable insights

4. QUALITY CHECKS:
   - Ensure logical consistency throughout
   - Verify that all key questions from the original query are addressed
   - Check that the conclusion follows from the presented evidence
   - Confirm the report provides value beyond a mere concatenation of agent outputs
   - Use concrete examples to support abstract concepts
   - Make sure the report is self-contained (readers shouldn't need to reference original agent outputs)

Your primary goal is to create a report that is:
1. Accurate and faithful to the information provided
2. Comprehensive in addressing the user's query
3. Clear and accessible to the intended audience
4. Actionable with specific recommendations when appropriate
5. Properly structured for easy understanding

Remember: Your job is to synthesize, not to add new information beyond what's in the agent outputs.
"""

    def process_messages(self, messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the messages and generate a comprehensive report.
        
        Args:
            messages: List of messages in the conversation
            state: Current state of the conversation
            metadata: Additional metadata
            
        Returns:
            Dict containing the report and routing metadata
        """
        try:
            # Extract original query
            original_query = ""
            for message in messages:
                if isinstance(message, dict) and message.get("type") == "human" and not original_query:
                    original_query = message.get("content", "")
                    break
                elif isinstance(message, HumanMessage) and not original_query:
                    original_query = message.content
                    break
            
            # Compile all relevant agent outputs
            agent_outputs = []
            research_findings = []
            planning_outputs = []
            coding_outputs = []
            browser_outputs = []
            
            # Process messages in chronological order to maintain narrative flow
            for message in messages:
                content = ""
                agent = None
                
                if isinstance(message, dict):
                    content = message.get("content", "")
                    # Try to get agent from metadata first
                    if isinstance(message.get("metadata"), dict):
                        agent = message.get("metadata", {}).get("agent")
                    
                    # If no agent in metadata, try to infer from content
                    if not agent and content:
                        if "RESEARCHER" in content or "Research findings" in content:
                            agent = "RESEARCHER"
                        elif "PLANNER" in content or "Plan:" in content:
                            agent = "PLANNER"
                        elif "CODER" in content or "```" in content:
                            agent = "CODER"
                        elif "BROWSER" in content or "browsing" in content.lower():
                            agent = "BROWSER"
                
                # Skip empty or irrelevant messages
                if not content or not agent:
                    continue
                
                # Categorize by agent type
                if agent == "RESEARCHER":
                    research_findings.append(content)
                elif agent == "PLANNER":
                    planning_outputs.append(content)
                elif agent == "CODER":
                    coding_outputs.append(content)
                elif agent == "BROWSER":
                    browser_outputs.append(content)
                
                # Add to overall outputs
                agent_outputs.append(f"--- {agent} OUTPUT ---\n{content}\n")
            
            # Compile the context for the report
            context = "\n\n".join(agent_outputs)
            
            # Check for empty context
            if not context.strip():
                # No agent outputs found, generate a minimal report based on the query
                return {
                    "content": f"""# Report on: {original_query}

## Executive Summary
I don't have sufficient information from other agents to generate a comprehensive report on this topic. It appears that the research and analysis phase may not have been completed.

## Recommendation
Please route this query back to the RESEARCHER agent to gather more information about the topic, or provide more specific questions to investigate.

""",
                    "metadata": {"agent": "REPORTER", "next": "RESEARCHER"}
                }
            
            # Create the prompt for generating the report
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", f"""Original Query: {original_query}

Agent Outputs:
{context}

Please synthesize this information into a comprehensive, well-structured report that answers the original query.
Organize the report by themes and insights, not by which agent provided the information.
Include ALL relevant information while eliminating redundancies.
Conclude with clear recommendations and next steps.
""")
            ])
            
            # Generate the report
            response = self.model.invoke(prompt)
            
            # Extract the content
            report_content = response.content if hasattr(response, 'content') else str(response)
            
            # Return the report with metadata
            return {
                "content": report_content,
                "metadata": {"agent": "REPORTER", "next": "FINISH"}
            }
            
        except Exception as e:
            logger.error(f"Error in reporter agent: {str(e)}")
            return {
                "content": f"""# Error in Report Generation

I encountered an error while attempting to compile the final report: {str(e)}

Here's what I could gather:

Original Query: {messages[0].get("content", "Unknown") if isinstance(messages[0], dict) else "Unknown"}

Unfortunately, I couldn't process all the agent outputs properly. This may indicate an issue with the data format or an internal processing error.

## Recommendation

Please route this back to the COORDINATOR to reassess the workflow.
""",
                "metadata": {"agent": "REPORTER", "next": "COORDINATOR"}
            }

# Create the reporter agent instance
reporter_agent = ReporterAgent()

def reporter_node(messages: List[Dict[str, Any]], state: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming messages in the reporter node.
    
    Args:
        messages: List of messages in the conversation
        state: Current state of the conversation
        metadata: Additional metadata
        
    Returns:
        Dict containing the report and routing metadata
    """
    return reporter_agent.process_messages(messages, state, metadata) 
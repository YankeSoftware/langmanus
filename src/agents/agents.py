from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    bash_tool,
    browser_tool,
    crawl_tool,
    python_repl_tool,
    tavily_tool,
    default_search_tool,
    memory_store_tool,
    memory_retrieve_tool,
)

from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
import logging

logger = logging.getLogger(__name__)

# Create researcher agent with conditional memory tools
researcher_tools = [default_search_tool, crawl_tool]
if memory_retrieve_tool:
    researcher_tools.append(memory_retrieve_tool)
    logger.info("Memory retrieval capability added to researcher agent")
else:
    logger.warning("Memory retrieval not available for researcher agent")

research_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    tools=researcher_tools,
    prompt=lambda state: apply_prompt_template("researcher", state),
)

coder_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["coder"]),
    tools=[python_repl_tool, bash_tool],
    prompt=lambda state: apply_prompt_template("coder", state),
)

browser_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["browser"]),
    tools=[browser_tool],
    prompt=lambda state: apply_prompt_template("browser", state),
)

# Add memory store capability to reporter if available
reporter_tools = []
if memory_store_tool:
    reporter_tools.append(memory_store_tool)
    logger.info("Memory storage capability added to reporter agent")
else:
    logger.warning("Memory storage not available for reporter agent")

reporter_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["reporter"]),
    tools=reporter_tools,
    prompt=lambda state: apply_prompt_template("reporter", state),
)

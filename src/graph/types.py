from typing import Literal, Dict, List, Optional, Union, Any
from typing_extensions import TypedDict
from langgraph.graph import MessagesState
from dataclasses import dataclass
from pydantic import BaseModel, Field

from src.config import TEAM_MEMBERS

# Define routing options
OPTIONS = TEAM_MEMBERS + ["FINISH"]


class Router(BaseModel):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: str = Field(description="The next node to run. For example, PLANNER, COORDINATOR, or FINISH")
    explanation: Optional[str] = Field(default=None, description="Explanation for why this node was chosen")


class State(MessagesState):
    """State for the agent workflow."""
    
    # Constants
    TEAM_MEMBERS: List[str]
    
    # Runtime Variables
    actions: str = ""  # Log of actions taken
    plan: str = ""     # The current plan
    tasks: str = ""    # Task list
    
    # Configuration
    deep_thinking_mode: bool = False
    search_before_planning: bool = False


@dataclass
class Command:
    """Command to update state and specify next node."""
    goto: str
    update: Optional[Dict[str, Any]] = None

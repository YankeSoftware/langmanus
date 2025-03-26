"""
Prompt templates for agent system.

This module provides structured, reusable prompts for the different agent roles
in the langmanus system. Templates are loaded from markdown files and can be
applied to state to generate system prompts for agents.
"""

from .template import get_prompt_template, apply_prompt_template, convert_message_to_dict

# Export core template functionality
__all__ = [
    "get_prompt_template", 
    "apply_prompt_template",
    "convert_message_to_dict"
]

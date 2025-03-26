import pytest
import os
import sys
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow import run_agent_workflow, enable_debug_logging, diagnose_environment
from src.config import TEAM_MEMBERS

# Configure logging to see debug messages during tests
logging.basicConfig(level=logging.DEBUG)

class TestEndToEndWorkflow:
    """
    End-to-end tests for the complete workflow system.
    
    These tests validate the full workflow with mocked components
    to ensure all parts work together properly.
    """
    
    @pytest.fixture
    def mock_llm_responses(self):
        """Mock LLM responses for different agents"""
        return {
            "researcher": "I've researched the topic and found the following key information...",
            "planner": "Here's a comprehensive plan to address the query...",
            "coordinator": "Based on the current state, we should route to: REPORTER",
            "coder": "Here's the implementation code for the solution...",
            "browser": "I've browsed the relevant websites and found...",
            "reporter": "Final report on the requested topic...",
            "supervisor": "Let's continue with the REPORTER to finalize the results."
        }
    
    @pytest.fixture
    def mock_user_input(self):
        """Return valid mock user input"""
        return "Create a summary of the latest AI research papers"
    
    @pytest.fixture
    def mock_graph(self, monkeypatch):
        """Mock the graph building and execution"""
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "Create a summary of the latest AI research papers"},
                {"role": "assistant", "content": "[RESEARCHER]: Here are my findings..."},
                {"role": "assistant", "content": "[PLANNER]: Here's the execution plan..."},
                {"role": "assistant", "content": "[REPORTER]: Final summary of AI research papers..."}
            ],
            "contributing_agents": ["RESEARCHER", "PLANNER", "REPORTER"],
            "execution_metrics": {
                "researcher": 1.5,
                "planner": 0.8,
                "reporter": 2.3
            }
        }
        
        # Patch the build_graph function
        monkeypatch.setattr("src.graph.build_graph", lambda: mock_graph)
        return mock_graph
    
    @pytest.fixture
    def mock_human_approval(self, monkeypatch):
        """Mock the human approval function to automatically approve"""
        monkeypatch.setattr("src.workflow.get_human_approval", lambda state: True)
    
    @pytest.fixture
    def mock_node_functions(self, monkeypatch, mock_llm_responses):
        """Mock all node functions to return predictable responses"""
        
        def mock_node_function(node_name):
            def _mock_node(messages, state, config):
                return {
                    "role": "assistant",
                    "content": f"[{node_name.upper()}]: {mock_llm_responses[node_name.lower()]}",
                    "metadata": {"agent": node_name.upper(), "next": "SUPERVISOR"}
                }
            return _mock_node
        
        # Mock each node function
        for agent in ["researcher", "planner", "coordinator", "coder", "browser", "reporter", "supervisor"]:
            monkeypatch.setattr(f"src.agents.{agent}.{agent}_node", mock_node_function(agent))
    
    def test_end_to_end_workflow_with_hitl(self, mock_graph, mock_human_approval, mock_user_input):
        """Test the full workflow execution with HITL enabled"""
        # Execute workflow
        result = run_agent_workflow(mock_user_input, debug=True, enable_hitl=True)
        
        # Verify the workflow was executed
        assert mock_graph.invoke.called
        
        # Verify the state contains the expected data
        assert "messages" in result
        assert "contributing_agents" in result
        assert len(result["messages"]) >= 3  # At least user input + 2 agent responses
        
        # Verify the execution_metrics are present
        assert "execution_metrics" in result
        assert len(result["execution_metrics"]) >= 2  # At least two agents contributed
    
    @patch("builtins.input", return_value="y")
    def test_workflow_with_real_hitl_input(self, mock_input, mock_graph, mock_node_functions, mock_user_input):
        """Test workflow with simulated user input for HITL"""
        # Execute workflow
        result = run_agent_workflow(mock_user_input, debug=True, enable_hitl=True)
        
        # Verify user was prompted for approval
        assert mock_input.called
        
        # Verify workflow completed
        assert "messages" in result
        assert len(result["contributing_agents"]) >= 2
    
    def test_workflow_without_hitl(self, mock_graph, mock_user_input):
        """Test workflow execution with HITL disabled"""
        # Execute workflow
        result = run_agent_workflow(mock_user_input, debug=True, enable_hitl=False)
        
        # Verify the workflow was executed directly
        assert mock_graph.invoke.called
        
        # No HITL process should have occurred
        assert result["hitl_enabled"] is False
    
    def test_workflow_with_api_config(self, mock_graph, mock_user_input):
        """Test workflow execution with custom API config"""
        # Custom API config
        api_config = {
            "model": "mistral-7b-instruct-v0.3",
            "temperature": 0.7,
            "max_tokens": 1500
        }
        
        # Execute workflow
        result = run_agent_workflow(mock_user_input, debug=True, api_config=api_config)
        
        # Verify API config was passed to the graph
        invoke_args = mock_graph.invoke.call_args[0][0]
        assert "api_config" in invoke_args
        assert invoke_args["api_config"]["model"] == "mistral-7b-instruct-v0.3"
        assert invoke_args["api_config"]["temperature"] == 0.7
    
    @pytest.mark.xfail(reason="Should fail with empty input")
    def test_workflow_with_empty_input(self):
        """Test that workflow fails properly with empty input"""
        with pytest.raises(ValueError, match="User input cannot be empty"):
            run_agent_workflow("")
    
    @patch("src.workflow.diagnose_environment")
    def test_diagnose_environment(self, mock_diagnose):
        """Test environment diagnostic function"""
        # Setup mock return value
        mock_diagnose.return_value = {
            "lm_studio_available": True,
            "openai_key_available": False,
            "memory_system_available": True
        }
        
        # Run diagnostic
        results = diagnose_environment(verbose=True)
        
        # Verify the diagnostic was called and returned expected structure
        assert mock_diagnose.called
        assert "lm_studio_available" in results
        assert "openai_key_available" in results 
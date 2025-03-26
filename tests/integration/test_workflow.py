import pytest
import os
import logging
from unittest.mock import patch, MagicMock
from src.workflow import run_agent_workflow, enable_debug_logging, diagnose_environment
from src.agents.base_agent import AgentError, AgentExecutionError, ModelAPIError


def test_enable_debug_logging():
    """Test that debug logging is properly enabled."""
    enable_debug_logging()
    logger = logging.getLogger("src")
    assert logger.getEffectiveLevel() == logging.DEBUG


@pytest.mark.skip(reason="Temporarily skipping this test")
def test_run_agent_workflow_basic():
    """Test basic workflow execution."""
    test_input = "What is the weather today?"
    result = run_agent_workflow(test_input)
    assert result is not None


def test_run_agent_workflow_empty_input():
    """Test workflow execution with empty input."""
    with pytest.raises(ValueError):
        run_agent_workflow("")


@pytest.mark.parametrize(
    "input_text,expected_exception",
    [
        ("", ValueError),
        (None, ValueError),
        ("   ", ValueError),
    ]
)
def test_run_agent_workflow_invalid_inputs(input_text, expected_exception):
    """Test workflow execution with various invalid inputs."""
    with pytest.raises(expected_exception):
        run_agent_workflow(input_text)


@patch('src.workflow.build_graph')
def test_run_agent_workflow_graph_error(mock_build_graph):
    """Test handling of errors in graph building."""
    # Setup mock to raise an exception
    mock_build_graph.side_effect = Exception("Graph build error")
    
    # Test error handling
    with pytest.raises(RuntimeError) as excinfo:
        run_agent_workflow("Test query")
    
    # Verify the error is properly wrapped
    assert "Workflow failed with error" in str(excinfo.value)
    assert "Graph build error" in str(excinfo.value)


@patch('src.workflow.get_human_approval')
def test_hitl_approval_workflow(mock_get_human_approval):
    """Test the HITL approval workflow."""
    # Mock the human approval to return True
    mock_get_human_approval.return_value = True
    
    # Create a simple initial state
    initial_state = {
        "messages": [{"role": "user", "content": "Test query"}],
        "hitl_enabled": True
    }
    
    # Apply the function to the test
    with patch('src.workflow.build_graph') as mock_build_graph:
        # Create a mock graph that returns the input state
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = initial_state
        mock_build_graph.return_value = mock_graph
        
        # Test with HITL enabled
        result = run_agent_workflow("Test query", enable_hitl=True)
        
        # Check that the HITL approval was called
        mock_get_human_approval.assert_called_once()
        
        # Verify that the workflow continued
        mock_graph.invoke.assert_called_once()


@patch('src.workflow.get_human_approval')
def test_hitl_rejection_workflow(mock_get_human_approval):
    """Test the HITL rejection workflow."""
    # Mock the human approval to return False
    mock_get_human_approval.return_value = False
    
    # Apply the function to the test
    with patch('src.agents.planner.planner_node') as mock_planner:
        mock_planner.return_value = {
            "content": "Test plan",
            "metadata": {}
        }
        
        # Test with HITL enabled
        result = run_agent_workflow("Test query", enable_hitl=True)
        
        # Check that the workflow was not continued
        assert result["hitl_approved"] is False


@patch('src.workflow.build_graph')
def test_api_error_handling(mock_build_graph):
    """Test handling of API errors."""
    # Setup mock to raise a ModelAPIError
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = ModelAPIError("API connection failed")
    mock_build_graph.return_value = mock_graph
    
    # Test error handling
    with pytest.raises(RuntimeError) as excinfo:
        run_agent_workflow("Test query")
    
    # Verify the error is properly wrapped with helpful message
    assert "Workflow failed with error" in str(excinfo.value)
    assert "API connection failed" in str(excinfo.value)


@patch('src.workflow.diagnose_environment')
def test_diagnose_environment(mock_diagnose):
    """Test the environment diagnostics function."""
    # Setup mock to return diagnostic results
    mock_diagnose.return_value = {
        "lm_studio": True,
        "env_variables": {
            "OPENAI_API_KEY": True,
            "BRAVE_API_KEY": False
        }
    }
    
    # Test the function
    results = diagnose_environment(verbose=True)
    
    # Verify structure of results
    assert isinstance(results, dict)
    assert "lm_studio" in results
    assert "env_variables" in results
    assert isinstance(results["env_variables"], dict)


def test_environment_variables_loading():
    """Test that environment variables are loaded correctly."""
    # Test with a temporary environment variable
    os.environ["TEST_ENV_VAR"] = "test_value"
    
    # Verify the variable is accessible
    assert os.environ.get("TEST_ENV_VAR") == "test_value"
    
    # Clean up
    del os.environ["TEST_ENV_VAR"]


@pytest.mark.parametrize(
    "hitl_env_var,expected_result",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("", False),  # Default to False for empty string
    ]
)
def test_hitl_environment_variable_parsing(hitl_env_var, expected_result):
    """Test that HITL environment variables are parsed correctly."""
    # Set the environment variable
    os.environ["HITL_ENABLED"] = hitl_env_var
    
    # Import the module to trigger environment variable reading
    from src.graph.builder import HITL_ENABLED
    
    # Verify the value
    assert HITL_ENABLED == expected_result
    
    # Clean up
    del os.environ["HITL_ENABLED"]

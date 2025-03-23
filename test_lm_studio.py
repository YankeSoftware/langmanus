#!/usr/bin/env python3
"""Test LM Studio compatibility with LangManus"""

import sys
import os
import logging
import argparse
import time
from src.workflow import run_agent_workflow, enable_debug_logging
from langchain_core.messages import HumanMessage
from src.llms.litellm_v2 import ChatLiteLLMV2
from src.utils.json_utils import parse_structured_output
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set environment variable for testing
os.environ["USE_LM_STUDIO"] = "true"

def test_basic_message_sending():
    """Test basic message sending to LM Studio."""
    logger.info("Testing basic message sending to LM Studio...")
    
    llm = ChatLiteLLMV2(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_base="http://localhost:1234/v1"
    )
    
    try:
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        response = llm.invoke(messages)
        logger.info(f"Successfully received response: {response}")
        return True
    except Exception as e:
        logger.error(f"Error during basic message test: {e}")
        return False

def test_complex_message_format():
    """Test handling of complex message formats."""
    logger.info("Testing complex message format handling...")
    
    llm = ChatLiteLLMV2(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_base="http://localhost:1234/v1"
    )
    
    try:
        # Test with system message and mixed formats
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": "I am an AI assistant."},
            {"role": "user", "content": "Tell me more about yourself."}
        ]
        response = llm.invoke(messages)
        logger.info(f"Successfully processed complex message format")
        return True
    except Exception as e:
        logger.error(f"Error during complex message test: {e}")
        return False

def test_json_structured_output():
    """Test structured JSON output with LM Studio."""
    logger.info("Testing structured JSON output with LM Studio...")
    
    llm = ChatLiteLLMV2(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_base="http://localhost:1234/v1"
    )
    
    try:
        # Define a simple schema
        class TestSchema(BaseModel):
            """Test schema for structured output."""
            message: str = Field(description="A test message")
            success: bool = Field(description="Whether the test was successful")
        
        # Test with structured output
        messages = [
            {"role": "user", "content": "Please respond with a JSON containing a message and success status."}
        ]
        
        # Use our enhanced ChatLiteLLMV2 with structured output
        chain = llm.with_structured_output(schema=TestSchema)
        response = chain.invoke(messages)
        
        logger.info(f"Successfully received structured response: {response}")
        return True
    except Exception as e:
        logger.error(f"Error during JSON structured output test: {e}")
        return False

def test_complex_json_schema():
    """Test handling of complex JSON schema."""
    logger.info("Testing complex JSON schema handling...")
    
    llm = ChatLiteLLMV2(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_base="http://localhost:1234/v1"
    )
    
    try:
        # Define a more complex schema
        class NestedItem(BaseModel):
            id: int = Field(description="Item ID")
            name: str = Field(description="Item name")
        
        class ComplexSchema(BaseModel):
            """Complex schema with nested objects and arrays."""
            title: str = Field(description="The title")
            items: list[NestedItem] = Field(description="A list of items")
            metadata: dict = Field(description="Additional metadata")
        
        # Test with structured output
        messages = [
            {"role": "user", "content": 
             "Please respond with a JSON object that has a title, a list of items (each with id and name), and a metadata dictionary."}
        ]
        
        # Use our enhanced ChatLiteLLMV2 with structured output and json_mode
        chain = llm.with_structured_output(schema=ComplexSchema, method="json_mode")
        response = chain.invoke(messages)
        
        logger.info(f"Successfully received complex schema response")
        return True
    except Exception as e:
        logger.error(f"Error during complex JSON schema test: {e}")
        return False

def test_manual_json_parsing():
    """Test our manual JSON parsing with potentially malformed responses."""
    logger.info("Testing manual JSON parsing capability...")
    
    from src.utils.json_utils import parse_structured_output
    
    test_cases = [
        # Well-formed JSON
        '{"message": "Test message", "success": true}',
        
        # JSON in code block
        '```json\n{"message": "Test message", "success": true}\n```',
        
        # JSON with single quotes
        "{'message': 'Test message', 'success': true}",
        
        # JSON with unquoted keys
        '{message: "Test message", success: true}',
        
        # JSON with trailing comma
        '{"message": "Test message", "success": true,}',
        
        # Nested JSON with formatting issues
        '{"items": [{"id": 1, "name": "Item 1"}, {"id": 2, name: "Item 2",}], "success": true}'
    ]
    
    class TestSchema(BaseModel):
        """Test schema for JSON parsing."""
        message: str = Field(default="")
        success: bool = Field(default=False)
    
    success = True
    for i, test_case in enumerate(test_cases):
        try:
            result = parse_structured_output(test_case, TestSchema)
            logger.info(f"Test case {i+1}: Successfully parsed: {result}")
        except Exception as e:
            logger.error(f"Test case {i+1}: Failed to parse: {e}")
            success = False
    
    return success

def test_performance():
    """Basic performance test to ensure reasonable response times."""
    logger.info("Testing performance with LM Studio...")
    
    llm = ChatLiteLLMV2(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        api_base="http://localhost:1234/v1"
    )
    
    try:
        start_time = time.time()
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        response = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        logger.info(f"Response time: {elapsed:.2f} seconds")
        if elapsed > 30:
            logger.warning("Response time is longer than expected (>30s)")
            # Still return True as this is just a warning
        
        return True
    except Exception as e:
        logger.error(f"Error during performance test: {e}")
        return False

def test_full_workflow():
    """Test the full agent workflow with LM Studio."""
    logger.info("Testing full agent workflow with LM Studio...")
    
    try:
        # Run a simple workflow
        result = run_agent_workflow(
            "Give me a short summary of quantum computing", 
            debug=True
        )
        logger.info("Workflow completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error during workflow test: {e}")
        return False

def main():
    """Run LM Studio compatibility tests."""
    parser = argparse.ArgumentParser(description="Test LM Studio compatibility")
    parser.add_argument("--full", action="store_true", help="Run full workflow test")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test", type=str, help="Run specific test by name")
    args = parser.parse_args()
    
    if args.debug:
        enable_debug_logging()
        logger.setLevel(logging.DEBUG)
    
    print("Testing LM Studio compatibility with LangManus...")
    
    # Define all tests
    all_tests = {
        "basic": ("Basic Message Test", test_basic_message_sending),
        "complex_message": ("Complex Message Format Test", test_complex_message_format),
        "json": ("JSON Structured Output Test", test_json_structured_output),
        "complex_json": ("Complex JSON Schema Test", test_complex_json_schema),
        "json_parsing": ("Manual JSON Parsing Test", test_manual_json_parsing),
        "performance": ("Performance Test", test_performance),
        "workflow": ("Full Workflow Test", test_full_workflow),
    }
    
    # Determine which tests to run
    tests_to_run = []
    
    if args.test:
        if args.test in all_tests:
            tests_to_run = [all_tests[args.test]]
        else:
            print(f"Test '{args.test}' not found. Available tests: {', '.join(all_tests.keys())}")
            return 1
    else:
        # Default set of tests
        tests_to_run = [
            all_tests["basic"],
            all_tests["complex_message"],
            all_tests["json"],
            all_tests["json_parsing"],
            all_tests["performance"],
        ]
        
        # Add full workflow test if requested
        if args.full:
            tests_to_run.append(all_tests["workflow"])
            tests_to_run.append(all_tests["complex_json"])
    
    # Run selected tests
    all_passed = True
    for name, test_func in tests_to_run:
        print(f"\nRunning {name}...")
        try:
            result = test_func()
            if result:
                print(f"✅ {name} PASSED")
            else:
                print(f"❌ {name} FAILED")
                all_passed = False
        except Exception as e:
            print(f"❌ {name} ERROR: {e}")
            all_passed = False
    
    # Print summary
    print("\n" + "="*50)
    if all_passed:
        print("✅ All tests PASSED! LM Studio compatibility looks good.")
    else:
        print("❌ Some tests FAILED. See logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
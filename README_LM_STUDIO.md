# Using LangManus with LM Studio

This guide will help you set up and run LangManus with LM Studio, a local LLM runner.

## Setup LM Studio

1. Download and install [LM Studio](https://lmstudio.ai/) for your platform.

2. In LM Studio:
   - Click "Add Model to Library" and select "Mistral-7B-Instruct-v0.3" or another compatible model
   - Once downloaded, select that model in your library
   - Click "Local Server" tab at the top
   - Enable "OpenAI Compatible Server"
   - Set port to 1234 (default)
   - Start the server by clicking "Start Server"

3. Verify the server is running by checking that you see "Server Status: Running" and the endpoint URL shows `http://localhost:1234/v1`

## Configure LangManus

1. Update your `.env` file with the following settings:
   ```
   BASIC_MODEL=mistralai/Mistral-7B-Instruct-v0.3
   BASIC_BASE_URL=http://localhost:1234/v1
   BASIC_API_KEY=not-needed

   REASONING_MODEL=mistralai/Mistral-7B-Instruct-v0.3
   REASONING_BASE_URL=http://localhost:1234/v1
   REASONING_API_KEY=not-needed
   ```

2. Run the LM Studio compatibility test:
   ```
   python test_lm_studio.py --debug
   ```

3. If the tests pass, you're ready to use LangManus with LM Studio!

## System Diagnostics

LangManus includes a comprehensive diagnostics tool to verify your setup:

```
python main.py --diagnostics
```

The diagnostics check will verify:
- LM Studio connectivity and available models
- Required environment variables
- Memory system availability
- RLHF (feedback) system availability

This is useful for troubleshooting and ensures all system components are functioning correctly before running your queries.

## Testing

### Basic Tests

To run all basic tests including message formatting, JSON parsing, and API communication:
```
python test_lm_studio.py --debug
```

### Specific Tests

To run a specific test (for troubleshooting):
```
python test_lm_studio.py --test basic         # Basic message formatting
python test_lm_studio.py --test json          # JSON structured output
python test_lm_studio.py --test json_parsing  # Manual JSON parsing
python test_lm_studio.py --test workflow      # Full workflow
```

### Full Testing Suite

To test the full workflow and all components:
```
python test_lm_studio.py --full --debug
```

## Example Usage

We've included a simple example to demonstrate LangManus with LM Studio:

```
python examples/lm_studio_basic.py
```

This interactive example allows you to send queries to the LangManus workflow using your LM Studio model.

## Technical Details

### Message Format Compatibility

LM Studio only supports the "user" and "assistant" roles. LangManus implements the following conversions:

1. System messages → User messages with role prefix
2. Function messages → Assistant messages with role prefix
3. Custom role messages → Assistant messages with role prefix

This ensures all messages sent to LM Studio are in a compatible format.

### JSON Structured Output

For structured output, LangManus uses three strategies:

1. **For models that support JSON mode:** Uses `response_format={"type": "json_schema"}` 
2. **For models that support function calling:** Uses OpenAI-style function calling
3. **For models with limited support (like Mistral):** 
   - Adds explicit JSON instructions in the prompt
   - Uses robust regex-based extraction to parse JSON from the response
   - Provides fallback values for failed parsing

### Performance Considerations

- Local models may run slower than cloud-based ones, especially on systems with limited GPU resources
- Complex workflows might take longer to complete
- For best performance:
  - Ensure you have a capable GPU with sufficient VRAM (8GB+ recommended)
  - Close other GPU-intensive applications while running LM Studio
  - Consider using smaller models (7B parameters or less) if you face performance issues

## Advanced Configuration

### Environment Variables

You can set these environment variables to control LM Studio integration:

```
USE_LM_STUDIO=true       # Force LM Studio compatibility mode
DEBUG_JSON_PARSING=true  # Enable verbose logging for JSON parsing
```

### Custom Model Configuration

You can modify the default agent-LLM mapping in `src/config/agents.py` to change which LLM is used for each role:

```python
AGENT_LLM_MAP = {
    "coordinator": "basic",
    "planner": "reasoning",
    "supervisor": "basic",
    "researcher": "basic",
    "coder": "basic",
    "browser": "vision",
    "reporter": "basic",
}
```

## Troubleshooting

### Common Issues

1. **"Cannot connect to LM Studio" error:**
   - Check if LM Studio server is running
   - Verify the port (default: 1234) is not blocked by firewall
   - Ensure the URL in .env matches your LM Studio server URL

2. **Invalid JSON Response errors:**
   - The model might be struggling to generate proper JSON
   - Try using a different model or adjusting the prompt
   - Enable debug logging to see the raw model output

3. **Slow performance:**
   - Check GPU utilization - LM Studio may be memory constrained
   - Try reducing model size or using a more efficient model

### Debugging

For detailed debugging information, add these flags:

```python
# Enable verbose logging
import logging
logging.getLogger("src").setLevel(logging.DEBUG)
logging.getLogger("litellm").setLevel(logging.DEBUG)
```

## Limitations

1. **Model Capabilities:** 
   - LM Studio with Mistral models doesn't natively support OpenAI's function/tool calling
   - Complex structured outputs might be less reliable than with models designed for this purpose

2. **Performance:** 
   - Local inference can be significantly slower than cloud-based services
   - Complex reasoning tasks might require more capable models (14B+ parameters)

3. **Prompt Sensitivity:**
   - Mistral models may be more sensitive to prompt formatting
   - You might need to adjust prompts for optimal results

## Contributing

If you encounter issues or have improvements for LM Studio compatibility, please:

1. Run the test suite with detailed logging: `python test_lm_studio.py --full --debug > debug_log.txt 2>&1`
2. Share your debug logs and environment details when reporting issues
3. Consider contributing fixes back to the project 
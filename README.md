# 🦜🤖 LangManus - AI Automation Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WeChat](https://img.shields.io/badge/WeChat-Langmanus-brightgreen?logo=wechat&logoColor=white)](./assets/wechat_community.jpg)
[![Discord Follow](https://dcbadge.vercel.app/api/server/m3MszDcn?style=flat)](https://discord.gg/m3MszDcn)

[English](./README.md) | [简体中文](./README_zh.md) | [日本語](./README_ja.md)

> Your intelligent AI system for complex tasks

This is a highly optimized version of LangManus, a community-driven AI automation framework that combines the power of language models with specialized tools and a simple memory system. Built upon the incredible work of the open source community, this version focuses on efficiency, reliability, and independence from third-party services.

## 🌟 Key Features

- **Multi-Agent Architecture**: Specialized agents for research, coding, planning, browsing, and reporting
- **Human-in-the-Loop (HITL)**: Get approval and provide feedback on plans before execution
- **LM Studio Integration**: Run completely locally with your preferred models
- **DeepSeek Integration**: Optional cloud-based LLM for complex reasoning tasks
- **Flexible Workflow**: Dynamic agent routing based on task requirements
- **Markdown Prompt Templates**: Easily customizable agent behavior
- **Memory System**: Track conversation context across sessions
- **Feedback System**: Provide ratings for continuous improvement

## 📋 Requirements

- Python 3.10+
- LM Studio (for local model execution) - [Download here](https://lmstudio.ai/)
- Internet connection (for web research and optional cloud LLMs)

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/langmanus.git
   cd langmanus
   ```

2. **Set up environment**:
   ```bash
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows
   .venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure LM Studio**:
   - Launch LM Studio
   - Load a compatible model (Mistral, Llama, etc.)
   - Enable the OpenAI compatible server (Settings > OpenAI API)
   - Ensure server is running on http://localhost:1234

4. **Set up environment variables**:
   ```bash
   # Copy example env file
   cp .env.example .env
   
   # Edit .env file with your API keys (optional)
   # BRAVE_API_KEY for web search
   # DEEPSEEK_API_KEY for cloud LLM
   ```

5. **Run LangManus**:
   ```bash
   # With HITL enabled (default)
   python main.py "Your complex query here"
   
   # Without HITL
   python main.py --no-hitl "Your complex query here"
   
   # Run diagnostics
   python main.py --diagnostics
   ```

## 🔍 Human-in-the-Loop Mode

LangManus includes HITL functionality that allows you to review and approve plans before execution:

1. Submit your query
2. The system performs initial research and generates a plan
3. You review the plan and can:
   - Approve (proceed with execution)
   - Reject (abort execution)
   - Modify (provide feedback and suggestions)
4. System executes with your guidance

Enable/disable using:
```bash
# Enable HITL (default)
python main.py --hitl "Your query"

# Disable HITL
python main.py --no-hitl "Your query" 
```

## 📐 Architecture

LangManus uses a graph-based workflow to coordinate multiple specialized agents:

- **Coordinator**: Manages overall workflow and delegates to specialists
- **Researcher**: Searches the web and gathers information
- **Planner**: Creates structured plans for solving problems
- **Coder**: Writes and debugs code
- **Browser**: Navigates web interfaces
- **Reporter**: Synthesizes findings into final reports
- **Supervisor**: Directs workflow between agents

## 🛠️ Customization

### Custom Prompts

Agent behavior is driven by markdown prompt templates in the `src/prompts` directory:
- Edit existing prompt files (*.md) to customize agent behavior
- System automatically loads these prompts at runtime

### Model Selection

Configure which models to use in your `.env` file:
- Use `LM_STUDIO_BASE_URL` and `LM_STUDIO_MODEL` for local models
- Set `DEEPSEEK_API_KEY` to enable DeepSeek for complex reasoning

## 🤝 Feedback System

Rate and improve agent responses:
- Rate responses on a scale of 1-5
- Provide textual feedback
- Helps improve the system over time

## 🐞 Troubleshooting

- **LM Studio Connection Issues**: Ensure LM Studio server is running and accessible at http://localhost:1234
- **Model Compatibility**: Use chat-tuned models like Mistral, Llama, etc.
- **Memory Issues**: For large models, ensure your system has sufficient RAM

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👥 Contributors

Special thanks to all the open source projects and contributors that make LangManus possible. We stand on the shoulders of giants.

In particular, we want to express our deep appreciation for:
- [LangChain](https://github.com/langchain-ai/langchain) for their exceptional framework that powers our LLM interactions and chains
- [LangGraph](https://github.com/langchain-ai/langgraph) for enabling our sophisticated multi-agent orchestration
- [Browser-use](https://pypi.org/project/browser-use/) for control browser

These amazing projects form the foundation of LangManus and demonstrate the power of open source collaboration.

## Memory System

LangManus includes a simple but effective memory system that enables:

1. **Context Persistence**: Maintain context between different sessions
2. **Knowledge Retention**: Store important discoveries or insights for future reference
3. **Pattern Recognition**: Recognize patterns in your interactions over time

### Memory Tools

LangManus includes specialized tools for memory interactions:

- `memory_store_tool`: Allows the Reporter agent to store important findings in memory
- `memory_retrieve_tool`: Enables the Researcher agent to access relevant memories

- `SimpleMemory` class: Core memory system that provides storage and retrieval functionality

## Advanced Features

The memory system in LangManus provides several advanced capabilities:

1. **Session Persistence**: Information from your conversations is maintained between sessions
2. **Keyword-Based Retrieval**: Find relevant memories based on keyword matching
3. **Tag-Based Organization**: Organize memories with customizable tags

By integrating this memory system with LangManus, we create a powerful system that:

1. Remembers past interactions and builds on them
2. Provides continuity between different sessions
3. Learns your preferences and patterns over time
4. Reduces repetitive explanations by maintaining context

## Acknowledgments

LangManus is built on the shoulders of giants. We'd like to thank:

- The [LangChain](https://github.com/langchain-ai/langchain) team for their incredible work
- [LM Studio](https://lmstudio.ai/) for providing local LLM inference
- The authors and maintainers of the Mistral-7B model
- All the contributors to this project

## LM Studio Integration

For detailed instructions on using LangManus with LM Studio for local LLM deployment, see:
[LM Studio Integration Guide](README_LM_STUDIO.md)

## System Diagnostics

Run comprehensive system checks to ensure all components are functioning:

```
python main.py --diagnostics
```

The diagnostics feature checks:
- LM Studio connectivity and available models
- Required environment variables
- Memory system availability
- RLHF (feedback) system availability

## Development

LangManus is designed with modularity in mind. The core components are:

- `src/graph`: Workflow definition and orchestration
- `src/llms`: LLM provider integrations
- `src/tools`: Tool implementations
- `src/integration`: Memory and feedback systems
- `src/utils`: Utility functions

# Langmanus - Advanced Multi-Agent System

LangManus is a powerful generalized agent system designed for autonomous research, planning, and problem-solving. It orchestrates a team of specialized AI agents that work together to tackle complex tasks with minimal human intervention.

## Key Features

- **Multi-Agent Workflow**: Coordinates specialized agents for research, planning, coding, and reporting
- **Autonomous Operation**: Manages state and transitions between agents to accomplish tasks independently
- **Loop Prevention**: Advanced loop detection and state tracking to ensure forward progress
- **Robust Research**: Multi-source information gathering with fallbacks for search failures
- **Dynamic Planning**: Strategic breakdowns of complex problems into manageable steps
- **Technical Implementation**: Code generation and system design capabilities
- **Comprehensive Reporting**: Clear synthesis of findings and actionable recommendations

## System Architecture

Langmanus employs a directed graph workflow with specialized agents:

- **Coordinator**: Entry point that analyzes queries and determines initial routing
- **Supervisor**: Monitors state, ensures progress, and makes routing decisions
- **Researcher**: Gathers information through web searches with fallback mechanisms
- **Planner**: Develops strategic approaches and implementation roadmaps
- **Coder**: Implements technical solutions with clean, well-structured code
- **Browser**: Simulates web interactions and navigation procedures
- **Reporter**: Synthesizes findings into comprehensive reports

## Getting Started

### Prerequisites

- Python 3.8+
- Required API keys:
  - OpenAI API key (for GPT-4o access)
  - Tavily API key (for search capabilities)
  - Google Search API key and CSE ID (optional backup search)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langmanus.git
   cd langmanus
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key"
   export TAVILY_API_KEY="your_tavily_api_key"
   export GOOGLE_API_KEY="your_google_api_key"  # Optional
   export GOOGLE_CSE_ID="your_google_cse_id"    # Optional
   ```

### Usage

Run the main interface:

```bash
python -m src.main
```

Send a query:

```python
from src.main import process_query

result = process_query("Research the capabilities and applications of the Enobio 32 from Neuroelectrics for 2025")
print(result)
```

## Example Workflows

### Research Workflow

1. User submits a query about a technical topic
2. Coordinator analyzes the query and routes to Researcher
3. Researcher performs comprehensive information gathering
4. Supervisor routes to Planner to develop an approach
5. Researcher gathers additional information as needed
6. Reporter synthesizes findings into a comprehensive report
7. Result is returned to the user

### Development Workflow

1. User submits a request to build a system
2. Coordinator analyzes the query and routes to Researcher
3. Researcher gathers background information
4. Planner develops a technical approach and architecture
5. Coder implements the solution with robust code
6. Reporter summarizes the implementation and provides usage instructions
7. Result is returned to the user

## Customization

You can customize agent behavior by modifying the system prompts in each agent's class:

```python
# Example: Customize the Researcher agent
from src.agents.researcher import research_agent

research_agent.system_prompt = """Your custom prompt here"""
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with LangChain for agent orchestration
- Powered by OpenAI's GPT-4o for advanced reasoning
- Tavily Search for robust information gathering

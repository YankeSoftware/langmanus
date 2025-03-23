# 🦜🤖 LangManus - AI Automation Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![WeChat](https://img.shields.io/badge/WeChat-Langmanus-brightgreen?logo=wechat&logoColor=white)](./assets/wechat_community.jpg)
[![Discord Follow](https://dcbadge.vercel.app/api/server/m3MszDcn?style=flat)](https://discord.gg/m3MszDcn)

[English](./README.md) | [简体中文](./README_zh.md) | [日本語](./README_ja.md)

> Your intelligent AI system for complex tasks

This is a highly optimized version of LangManus, a community-driven AI automation framework that combines the power of language models with specialized tools and a simple memory system. Built upon the incredible work of the open source community, this version focuses on efficiency, reliability, and independence from third-party services.

## Key Features

LangManus includes several important features:

- **Local LLM Integration**: Uses [LM Studio](https://lmstudio.ai/) for local model inference
- **Privacy-Focused Search**: Uses Brave Search API, prioritizing privacy
- **Simple Memory System**: Built-in memory system for retaining context between sessions
- **Complete Local Operation**: All components can run locally for maximum privacy and control
- **Multi-agent Workflows**: Orchestrated team of specialized agents to tackle complex tasks
- **Memory Integration**: Context-aware conversations with memory capabilities
- **Tool Integration**: Web search, coding, and other capabilities
- **System Diagnostics**: Comprehensive checks for all system components
- **RLHF Integration**: Collect and utilize user feedback to improve responses

## Demo

**Task**: Calculate the influence index of DeepSeek R1 on HuggingFace. This index can be designed using a weighted sum of factors such as followers, downloads, and likes.

**LangManus's Fully Automated Plan and Solution**:
1. Gather the latest information about "DeepSeek R1", "HuggingFace", and related topics through online searches.
2. Interact with a Chromium instance to visit the HuggingFace official website, search for "DeepSeek R1" and retrieve the latest data, including followers, likes, downloads, and other relevant metrics.
3. Find formulas for calculating model influence using search engines and web scraping.
4. Use Python to compute the influence index of DeepSeek R1 based on the collected data.
5. Present a comprehensive report to the user.

![Demo](./assets/demo.gif)

- [View on YouTube](https://youtu.be/sZCHqrQBUGk)

## Table of Contents

- [Quick Start](#quick-start)
- [Project Statement](#project-statement)
- [Architecture](#architecture)
- [Features](#features)
- [Why LangManus?](#why-langmanus)
- [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Configuration](#configuration)
- [Usage](#usage)
- [Docker](#docker)
- [Web UI](#web-ui)
- [Development](#development)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Quick Start

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/langmanus.git
   cd langmanus
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   BRAVE_API_KEY=your_brave_search_key
   
   # For LM Studio setup
   BASIC_MODEL=mistralai/Mistral-7B-Instruct-v0.3
   BASIC_BASE_URL=http://localhost:1234/v1
   BASIC_API_KEY=not-needed
   
   REASONING_MODEL=mistralai/Mistral-7B-Instruct-v0.3
   REASONING_BASE_URL=http://localhost:1234/v1
   REASONING_API_KEY=not-needed
   ```

4. Launch the system:
   ```
   # On Windows
   launch.bat
   
   # On Linux/Mac
   chmod +x launch.sh
   ./launch.sh
   ```

5. Or run directly with Python:
   ```
   python main.py "your query here"
   ```

## Project Statement

This is an academically driven open-source project, developed by a group of former colleagues in our spare time. It aims to explore and exchange ideas in the fields of Multi-Agent and DeepResearch.

- **Purpose**: The primary purpose of this project is academic research, participation in the GAIA leaderboard, and the future publication of related papers.
- **Independence Statement**: This project is entirely independent and unrelated to our primary job responsibilities. It does not represent the views or positions of our employers or any organizations.
- **No Association**: This project has no association with Manus (whether it refers to a company, organization, or any other entity).
- **Clarification Statement**: We have not promoted this project on any social media platforms. Any inaccurate reports related to this project are not aligned with its academic spirit.
- **Contribution Management**: Issues and PRs will be addressed during our free time and may experience delays. We appreciate your understanding.
- **Disclaimer**: This project is open-sourced under the MIT License. Users assume all risks associated with its use. We disclaim any responsibility for any direct or indirect consequences arising from the use of this project.

## 项目声明

本项目是一个学术驱动的开源项目，由一群前同事在业余时间开发，旨在探索和交流 Multi-Agent 和 DeepResearch 相关领域的技术。

- **项目目的**：本项目的主要目的是学术研究、参与 GAIA 排行榜，并计划在未来发表相关论文。
- **独立性声明**：本项目完全独立，与我们的本职工作无关，不代表我们所在公司或任何组织的立场或观点。
- **无关联声明**：本项目与 Manus（无论是公司、组织还是其他实体）无任何关联。
- **澄清声明**：我们未在任何社交媒体平台上宣传过本项目，任何与本项目相关的不实报道均与本项目的学术精神无关。
- **贡献管理**：Issue 和 PR 将在我们空闲时间处理，可能存在延迟，敬请谅解。
- **免责声明**：本项目基于 MIT 协议开源，使用者需自行承担使用风险。我们对因使用本项目产生的任何直接或间接后果不承担责任。

## Architecture

LangManus implements a hierarchical multi-agent system where a supervisor coordinates specialized agents to accomplish complex tasks:

![LangManus Architecture](./assets/architecture.png)

The system consists of the following agents working together:

1. **Coordinator** - The entry point that handles initial interactions and routes tasks
2. **Planner** - Analyzes tasks and creates execution strategies
3. **Supervisor** - Oversees and manages the execution of other agents
4. **Researcher** - Gathers and analyzes information using Brave Search and retrieved personal memories
5. **Coder** - Handles code generation and modifications
6. **Browser** - Performs web browsing and information retrieval
7. **Reporter** - Generates reports and stores important information in your personalized memory system

## Features

### Core Capabilities

- 🤖 **LLM Integration**
    - Support for open source models like Qwen
    - OpenAI-compatible API interface
    - Multi-tier LLM system for different task complexities

### Tools and Integrations

- 🔍 **Search and Retrieval**
    - Web search via Brave Search API
    - Neural search with Jina
    - Advanced content extraction

### Development Features

- 🐍 **Python Integration**
    - Built-in Python REPL
    - Code execution environment
    - Package management with uv

### Workflow Management

- 📊 **Visualization and Control**
    - Workflow graph visualization
    - Multi-agent orchestration
    - Task delegation and monitoring

## Why LangManus?

We believe in the power of open source collaboration. This project wouldn't be possible without the amazing work of projects like:

- [Qwen](https://github.com/QwenLM/Qwen) for their open source LLMs
- [Brave Search](https://brave.com/search/) for search capabilities
- [Jina](https://jina.ai/) for crawl search technology
- [Browser-use](https://pypi.org/project/browser-use/) for control browser
- And many other open source contributors

We're committed to giving back to the community and welcome contributions of all kinds - whether it's code, documentation, bug reports, or feature suggestions.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- [LM Studio](https://lmstudio.ai/) - For running local LLMs
- [Brave Search API Key](https://brave.com/search/api/) - For search functionality

### Installation

LangManus leverages [uv](https://github.com/astral-sh/uv) as its package manager to streamline dependency management:

```bash
# Clone both repositories
git clone https://github.com/[your-username]/langmanus.git
cd langmanus

# Install dependencies
uv sync

# Playwright install to use Chromium for browser-use by default
uv run playwright install

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

This personalized version uses the following environment variables:

```ini
# LM Studio Configuration for local inference
REASONING_API_KEY=your_api_key
REASONING_BASE_URL=http://localhost:1234/v1
REASONING_MODEL=openai/deepseek-r1:7b

BASIC_API_KEY=your_api_key
BASIC_BASE_URL=http://localhost:1234/v1
BASIC_MODEL=openai/deepseek-r1:7b

VL_API_KEY=your_api_key
VL_BASE_URL=http://localhost:1234/v1
VL_MODEL=openai/deepseek-r1:7b

# Brave Search API Key (required)
BRAVE_API_KEY=your_brave_search_api_key
```

### Usage

First, ensure LM Studio is running with the appropriate models loaded. Then:

```bash
# Run the personalized AI system
uv run main.py
```

For the API server with streaming support:

```bash
# Start the API server
uv run server.py
```

## Advanced Configuration

### Memory System Configuration

The simple memory system is automatically integrated and will store memories in the simple memory system repository. You can customize the memory system by modifying the `src/integration/memory.py` file.

### Search Configuration

This version uses Brave Search exclusively. You can adjust the number of search results in `src/config/tools.py`:

```python
# Brave Search configuration
BRAVE_MAX_RESULTS = 5  # Adjust as needed
```

### Agent Prompts

The agent prompts have been updated to be aware of the memory system:

- **Researcher** (`src/prompts/researcher.md`): Retrieves relevant memories before searching
- **Reporter** (`src/prompts/reporter.md`): Stores important information in the memory system

## Docker

LangManus can be run in a Docker container. default serve api on port 8000.

Before run docker, you need to prepare environment variables in `.env` file.

```bash
docker build -t langmanus .
docker run --name langmanus -d --env-file .env -e CHROME_HEADLESS=True -p 8000:8000 langmanus
```

You can also just run the cli with docker.

```bash
docker build -t langmanus .
docker run --rm -it --env-file .env -e CHROME_HEADLESS=True langmanus uv run python main.py
```

## Web UI

LangManus provides a default web UI.

Please refer to the [langmanus/langmanus-web-ui](https://github.com/langmanus/langmanus-web) project for more details.

## Development

### Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test file
pytest tests/integration/test_workflow.py

# Run with coverage
make coverage
```

### Code Quality

```bash
# Run linting
make lint

# Format code
make format
```

## FAQ

Please refer to the [FAQ.md](docs/FAQ.md) for more details.

## Contributing

We welcome contributions of all kinds! Whether you're fixing a typo, improving documentation, or adding a new feature, your help is appreciated. Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to get started.

## License

This project is open source and available under the [MIT License](LICENSE).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=langmanus/langmanus&type=Date)](https://www.star-history.com/#langmanus/langmanus&Date)

## Acknowledgments

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

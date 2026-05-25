# MCP Basic Projects

Hands-on learning path through **MCP (Model Context Protocol)** — each project builds one core concept so you actually _write_ servers, clients, and agents, not just read about them.

## What is MCP (in 30 seconds)

MCP is a protocol that lets AI apps (clients) discover and call tools exposed by external services (servers) — like a USB-C port for AI tools.

```
┌──────────────┐       MCP Protocol        ┌──────────────┐
│  MCP Client  │  ◄───────────────────────► │  MCP Server  │
│ (Claude, AI  │    tools / resources /     │ (your code   │
│   agent)     │    prompts / sampling      │  exposing     │
└──────────────┘                            │  functions)  │
                                            └──────────────┘
```

**Three things a server can expose:**
- **Tools** — functions the LLM can call (e.g. `get_weather`, `query_db`)
- **Resources** — read-only data the LLM can access (e.g. files, DB rows)
- **Prompts** — reusable prompt templates with arguments

**Two transport modes:**
- **stdio** — server runs as a child process, communicates over stdin/stdout (local)
- **streamable-http** — server runs as an HTTP service (local or remote)

## Projects

| # | File | What You Build | Concepts Covered |
|---|------|---------------|-----------------|
| 1 | `weather_agent.py` | LangGraph agent using a community MCP server | `StdioServerParameters`, `ClientSession`, `stdio_client`, `load_mcp_tools`, child process lifecycle, `create_react_agent` |

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# 4. Run a project
python 01_hello_mcp_server.py
```

## MCP Dev Tools (your best friends)

```bash
# Inspector UI — test any server interactively in the browser
mcp dev 01_hello_mcp_server.py

# Inspect — see what tools/resources/prompts a server exposes
mcp inspect 01_hello_mcp_server.py

# Install into Claude Desktop — register your server so Claude can use it
mcp install 01_hello_mcp_server.py
```

## Prerequisites

- Python 3.10+
- An Anthropic API key (for agent projects)
- Node.js 18+ (only for `mcp dev` inspector UI)
- Basic Python + async/await knowledge

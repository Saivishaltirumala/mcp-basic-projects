# ============================================================
# weather_agent.py — LangGraph Agent + MCP STDIO Transport
# ============================================================
#
# GOAL: Understand how the MCP STDIO transport works.
#
# We do NOT write an MCP server. We use an existing community
# server (@h1deya/mcp-server-weather) that wraps the US
# National Weather Service API.
#
#
# HOW IT WORKS (the big picture):
#
#   ┌────────────────────┐  stdin (JSON-RPC)  ┌──────────────────┐
#   │  This script       │ ──────────────────► │  npx weather     │
#   │  (MCP Host/Client) │                     │  server (child)  │
#   │                    │ ◄────────────────── │                  │
#   └────────────────────┘  stdout (JSON-RPC)  └──────────────────┘
#                                                      │
#                                                      │ HTTPS
#                                                      ▼
#                                               ┌──────────────────┐
#                                               │  api.weather.gov │
#                                               │  (NWS — free,    │
#                                               │   no API key)    │
#                                               └──────────────────┘
#
#   1. We spawn `npx mcp-server-weather` as a child process.
#   2. We talk to it over stdin/stdout — that's the "stdio transport".
#   3. The child fetches live weather data from api.weather.gov.
#   4. Results flow back through stdout into our LangGraph agent.
#
#
# WHEN TO USE STDIO vs STREAMABLE-HTTP:
# ──────────────────────────────────────
#   STDIO (this project):
#     - Server runs LOCALLY on your machine as a child process.
#     - Your script spawns it, talks via stdin/stdout, kills it when done.
#     - Cannot be shared — only YOUR machine can use it.
#     - Example: npx mcp-server-weather running on your laptop.
#
#   STREAMABLE-HTTP (for cloud/remote):
#     - Server runs on a REMOTE machine (cloud, VPS, HF Spaces, etc.).
#     - Anyone with the URL can connect to it over the internet.
#     - Server stays running independently — your script doesn't spawn it.
#     - Example: https://your-app.hf.space/mcp
#
#   Quick rule: Local server? → stdio. Cloud server? → streamable-http.
#
#
# IMPORTANT — This server is US-only!
#   The NWS API only covers US locations. Query 1 tests this
#   limitation on purpose. Query 2 uses a US city.
#
#
# ENV SETUP:
#   The weather server needs NO API key (NWS is free).
#   You only need an Anthropic key for the LLM:
#     ANTHROPIC_API_KEY=sk-ant-...   (in your .env file)
#
# PREREQUISITES:
#   - Node.js 18+ (for npx)
#   - pip install -r requirements.txt
#
# RUN:
#   python weather_agent.py
#
# ============================================================

import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


async def main():

    # ===========================================================
    # STEP 1: Configure the MCP server connection
    # ===========================================================
    #
    # This is just a config dict — nothing starts yet.
    #   - "weather"   → a label we pick (can be anything)
    #   - "command"   → what program to run
    #   - "args"      → arguments for npx ("-y" = auto-install)
    #   - "transport" → "stdio" = talk via stdin/stdout pipes
    #                   (use "stdio" for local servers that run on your machine,
    #                    use "streamable_http" for remote/cloud servers)
    #
    # If a server needed an API key, you'd add:
    #   "env": {"API_KEY": "your-key-here"}

    server_config = {
        "weather": {
            "command": "npx",
            "args": ["-y", "@h1deya/mcp-server-weather"],
            "transport": "stdio",
        }
    }

    print("=" * 60)
    print("  MCP STDIO TRANSPORT DEMO")
    print("=" * 60)
    print(f"\n  Server command : npx -y @h1deya/mcp-server-weather")
    print(f"  Transport      : stdio (stdin/stdout pipes)")
    print(f"  API key needed : None (NWS is free)")

    # ===========================================================
    # STEP 2: Create client and discover tools
    # ===========================================================
    #
    # MultiServerMCPClient(config) just stores the config — no
    # connection happens yet.
    #
    # await client.get_tools() is where the magic happens:
    #   1. Spawns `npx mcp-server-weather` as a child process
    #   2. Connects to its stdin/stdout pipes
    #   3. Does the MCP handshake
    #   4. Asks the server "what tools do you have?"
    #   5. Wraps the response as LangChain Tool objects
    #
    # Later, when the agent calls a tool, the adapter opens
    # a NEW connection to the child process each time.
    # (spawn → handshake → call tool → get result → close)

    client = MultiServerMCPClient(server_config)
    tools = await client.get_tools()

    print(f"\n  Tools discovered from server:")
    for tool in tools:
        print(f"    - {tool.name}: {tool.description[:80]}")

    # ===========================================================
    # STEP 3: Build the LangGraph ReAct agent
    # ===========================================================
    #
    # create_react_agent builds a loop:
    #   User question → LLM picks a tool → calls MCP server
    #   → gets result → LLM answers (or calls another tool)

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=1024,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
    )

    # ===========================================================
    # STEP 4: Run queries
    # ===========================================================

    # ── Query 1: Hyderabad (India) ────────────────────────
    # Will fail — NWS only covers US locations.
    # Shows how the agent handles tool limitations.

    print("\n" + "=" * 60)
    print("  QUERY 1: What is the current weather in Hyderabad?")
    print("  (Testing: non-US location -> server limitation)")
    print("=" * 60)

    result1 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the current weather in Hyderabad?"}]}
    )

    print(f"\n  Agent's answer:\n  {result1['messages'][-1].content}")

    # ── Query 2: San Francisco (US) ───────────────────────
    # Full working flow: agent calls get-forecast with SF
    # coordinates → NWS returns data → agent summarizes.

    print("\n\n" + "=" * 60)
    print("  QUERY 2: What is the weather forecast for San Francisco?")
    print("  (Testing: US location -> full working flow)")
    print("=" * 60)

    result2 = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the weather forecast for San Francisco?"}]}
    )

    print(f"\n  Agent's answer:\n  {result2['messages'][-1].content}")

    print("\n\n" + "=" * 60)
    print("  Done! MCP server shut down cleanly.")
    print("=" * 60)


# asyncio.run() starts the async event loop and runs main().
# Needed because `async with` only works inside async functions.
if __name__ == "__main__":
    asyncio.run(main())

# ============================================================
# music_agent.py — LangGraph Agent + MCP Streamable HTTP Transport
# ============================================================
#
# GOAL: Understand how MCP works with a REMOTE cloud server.
#
# In weather_agent.py (Project 1), we used STDIO transport:
#   - Server ran LOCALLY as a child process on your machine.
#   - Communication happened via stdin/stdout pipes.
#   - When your script exited, the server died.
#
# Here we use STREAMABLE HTTP transport:
#   - Server is ALREADY RUNNING in the cloud (Hugging Face Spaces).
#   - Communication happens via HTTP requests over the internet.
#   - Server stays alive even after your script exits.
#
#
# HOW IT WORKS:
#
#   ┌──────────────────┐       HTTPS         ┌──────────────────────┐
#   │  This script      │ ──────────────────► │  HF Spaces           │
#   │  (your laptop)    │                     │  (Indian Music MCP   │
#   │                   │ ◄────────────────── │   server, always on) │
#   └──────────────────┘    SSE stream        └──────────────────────┘
#                                                      │
#          Your request goes over the internet         │ HTTPS
#          (not local pipes like stdio)                ▼
#                                               ┌──────────────────┐
#                                               │  iTunes India API │
#                                               └──────────────────┘
#
#   The client sends requests via HTTP POST.
#   The server sends responses back via SSE (Server-Sent Events).
#   Together this is called "Streamable HTTP" transport in MCP.
#
#
# SSE vs STREAMABLE HTTP — WHAT'S THE DIFFERENCE?
# ────────────────────────────────────────────────
#   SSE (Server-Sent Events) is a web standard for one-way streaming
#   from server → client. MCP originally had a pure "SSE transport"
#   that combined SSE + HTTP POST for two-way communication.
#
#   "Streamable HTTP" is the newer, improved version that ALSO uses
#   SSE under the hood but is more robust. It replaced the old SSE
#   transport in MCP.
#
#   Quick rule:
#     - Old MCP servers → transport: "sse" (endpoint usually /sse)
#     - New MCP servers → transport: "streamable_http" (endpoint usually /mcp)
#     - Local servers   → transport: "stdio" (no URL, runs as child process)
#
#
# HOW THIS DIFFERS FROM STDIO (weather_agent.py):
# ────────────────────────────────────────────────
#   STDIO (Project 1):
#     - Config needs: command, args (what to run locally)
#     - Your script SPAWNS the server as a child process
#     - Talks via stdin/stdout (instant, no network)
#     - Server dies when your script exits
#
#   STREAMABLE HTTP (this project):
#     - Config needs: url (where the server lives on the internet)
#     - Server is ALREADY RUNNING — you just connect to it
#     - Talks via HTTP + SSE (over the internet, has latency)
#     - Server stays alive after your script exits
#     - Need to handle: timeouts, cold starts, network errors
#
#
# ENV SETUP:
#   ANTHROPIC_API_KEY=sk-ant-...   (in your .env file)
#   No key needed for the music server — it's a public HF Space.
#
# RUN:
#   python music_agent.py
#
# ============================================================

import asyncio
from datetime import timedelta

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
    # Compare this with the STDIO config from weather_agent.py:
    #
    #   STDIO (local):                      STREAMABLE HTTP (remote):
    #   {                                   {
    #     "command": "npx",                   "url": "https://....hf.space/mcp",
    #     "args": ["-y", "mcp-server-..."],   "transport": "streamable_http",
    #     "transport": "stdio",               "timeout": timedelta(seconds=60),
    #   }                                   }
    #
    # Key difference: no command/args needed — the server is already
    # running in the cloud. We just point to its URL.

    server_config = {
        "indian-music": {
            "url": "https://saivishaltirumala-indian-music-charts-mcp-server.hf.space/mcp",
            "transport": "streamable_http",

            # timeout: How long to wait for the server to respond.
            # HF Spaces can take up to 60s on cold start (when the
            # Space has been idle and needs to wake up).
            "timeout": timedelta(seconds=60),

            # sse_read_timeout: How long to keep the SSE stream open
            # waiting for data. Tool calls that fetch external data
            # (like iTunes charts) can take a few seconds.
            "sse_read_timeout": timedelta(seconds=300),
        }
    }

    print("=" * 60)
    print("  MCP STREAMABLE HTTP TRANSPORT DEMO")
    print("=" * 60)
    print(f"\n  Server URL     : ...indian-music-charts-mcp-server.hf.space/mcp")
    print(f"  Transport      : streamable_http (HTTP + SSE over internet)")
    print(f"  API key needed : None (public HF Space)")

    # ===========================================================
    # STEP 2: Create client and discover tools
    # ===========================================================
    #
    # Same two lines as weather_agent.py — the only difference
    # is the config above (url instead of command/args).
    #
    # Behind the scenes, instead of spawning a child process,
    # the adapter sends an HTTP POST to the HF Space URL and
    # reads the tool list from the SSE response stream.

    client = MultiServerMCPClient(server_config)

    try:
        tools = await client.get_tools()
    except Exception as e:
        print(f"\n  ERROR: Could not connect to the remote server.")
        print(f"  {e}")
        print(f"\n  The HF Space might be sleeping. Wait 30s and try again.")
        return

    print(f"\n  Tools discovered from server:")
    for tool in tools:
        print(f"    - {tool.name}: {tool.description[:80]}")

    # ===========================================================
    # STEP 3: Build the LangGraph ReAct agent
    # ===========================================================

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

    # ── Query 1 ───────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  QUERY 1: What are the top trending Bollywood songs?")
    print("=" * 60)

    try:
        result1 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What are the top trending Bollywood songs right now?"}]}
        )
        print(f"\n  Agent's answer:\n  {result1['messages'][-1].content}")

    # These errors DON'T exist with STDIO (local server = no network).
    # With a remote server, you must handle network problems.
    except TimeoutError:
        print("\n  ERROR: Server took too long. HF Space might be cold-starting.")
    except ConnectionError:
        print("\n  ERROR: Could not reach the server. Check your internet.")

    # ── Query 2 ───────────────────────────────────────────────

    print("\n\n" + "=" * 60)
    print("  QUERY 2: What are the top Telugu songs?")
    print("=" * 60)

    try:
        result2 = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Show me the top Telugu songs"}]}
        )
        print(f"\n  Agent's answer:\n  {result2['messages'][-1].content}")

    except TimeoutError:
        print("\n  ERROR: Server took too long. HF Space might be cold-starting.")
    except ConnectionError:
        print("\n  ERROR: Could not reach the server. Check your internet.")

    print("\n\n" + "=" * 60)
    print("  Done!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

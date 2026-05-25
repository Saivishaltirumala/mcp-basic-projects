# ============================================================
# app.py — Streamlit Chat + MCP Streamable HTTP Transport
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
#   │  This Streamlit   │ ──────────────────► │  HF Spaces           │
#   │  app (your laptop)│                     │  (Indian Music MCP   │
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
#   Your Indian Music server uses streamable-http (the modern one).
#   So we use transport: "streamable_http" in the config below.
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
#   streamlit run app.py
#
# ============================================================

import asyncio
from datetime import timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic


# ── MCP Server Config ────────────────────────────────────────
#
# Compare this with the STDIO config from weather_agent.py:
#
#   STDIO (local):                         STREAMABLE HTTP (remote):
#   {                                      {
#     "command": "npx",                      "url": "https://....hf.space/mcp",
#     "args": ["-y", "mcp-server-..."],      "transport": "streamable_http",
#     "transport": "stdio",                  "timeout": timedelta(seconds=60),
#   }                                      }
#
#   No command/args needed — the server    No spawning needed — the server
#   is already running in the cloud.       is already running in the cloud.

SERVER_CONFIG = {
    "indian-music": {
        "url": "https://saivishaltirumala-indian-music-charts-mcp-server.hf.space/mcp",
        "transport": "streamable_http",

        # timeout: How long to wait for the server to respond to
        # a single HTTP request. HF Spaces can take up to 60s on
        # cold start (when the Space has been idle and needs to wake up).
        "timeout": timedelta(seconds=60),

        # sse_read_timeout: How long to keep the SSE stream open
        # waiting for data. Tool calls that fetch external data
        # (like iTunes charts) can take a few seconds.
        "sse_read_timeout": timedelta(seconds=300),
    }
}


# ── Agent Logic ──────────────────────────────────────────────

async def ask_agent(question: str, chat_history: list[dict]) -> str:
    """Send a question to the LangGraph agent and return the answer.

    For EACH call, the adapter:
      1. Opens an HTTP connection to the HF Space
      2. Does the MCP handshake
      3. Discovers tools (list_genres, get_top_songs, etc.)
      4. Runs the agent (LLM decides → calls tool via HTTP → gets result)
      5. Closes the connection

    This is different from STDIO where step 1 would SPAWN a process.
    Here it just opens an HTTP connection — much faster, no process management.
    """

    # Create client — just stores the config, no connection yet.
    client = MultiServerMCPClient(SERVER_CONFIG)

    # Discover tools from the remote server.
    # This sends an HTTP request to the HF Space and asks
    # "what tools do you have?" — the server responds with
    # tool names, descriptions, and parameter schemas.
    tools = await client.get_tools()

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=1024,
    )

    agent = create_react_agent(model=llm, tools=tools)

    # Build the message list from chat history + new question.
    # This gives the agent context from the conversation so it
    # can handle follow-ups like "now show me Tamil songs".
    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in chat_history
    ]
    messages.append({"role": "user", "content": question})

    result = await agent.ainvoke({"messages": messages})

    return result["messages"][-1].content


# ── Streamlit UI ─────────────────────────────────────────────

st.set_page_config(page_title="Indian Music Charts Agent", page_icon="🎵")
st.title("🎵 Indian Music Charts Agent")
st.caption(
    "Powered by a remote MCP server on Hugging Face Spaces — "
    "ask about trending Indian music across Bollywood, Telugu, Tamil, Punjabi & more."
)

# Initialize chat history in session state.
# st.session_state persists across Streamlit reruns (each button
# click, each user message triggers a full script rerun).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input.
if prompt := st.chat_input("Ask about Indian music charts..."):

    # Add user message to history and display it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response.
    with st.chat_message("assistant"):
        with st.spinner("Connecting to MCP server on HF Spaces..."):

            try:
                response = asyncio.run(
                    ask_agent(prompt, st.session_state.messages[:-1])
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

            # ── Error handling for remote server issues ────────
            #
            # These errors DON'T exist with STDIO transport because
            # the server runs locally. With a remote server, you must
            # handle network problems:

            except TimeoutError:
                # HF Space might be cold-starting (waking up from idle).
                # First request after inactivity can take 30-60 seconds.
                st.error(
                    "⏱️ The server took too long to respond. "
                    "The HF Space might be waking up from a cold start — "
                    "wait 30 seconds and try again."
                )

            except ConnectionError:
                # Network issues — DNS failure, server down, etc.
                st.error(
                    "🔌 Could not connect to the MCP server. "
                    "Check your internet connection or verify the HF Space is running."
                )

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")

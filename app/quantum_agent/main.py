"""
AgentCore entrypoint for the Strands quantum optimization agent.

Connects to the MCP server via SigV4-signed Streamable HTTP using
aws_iam_streamablehttp_client from mcp_proxy_for_aws, then runs the
Strands agent backed by Amazon Bedrock Nova 2 Lite.

Environment variables
---------------------
MCP_URL   : Required. Full URL of the MCP server endpoint,
            e.g. https://<agentcore-endpoint>/mcp
AWS_REGION: AWS region for Bedrock and SigV4 signing.
            Defaults to us-east-1.

Performance design
------------------
BedrockModel, MCPClient, the MCP connection, and the filtered tool list
are all initialized once at first invocation and reused for the lifetime
of the process.  handler() pays only the cost of building the per-request
session manager and running the agent call.

Streaming design
----------------
handler() is an async generator.  It yields SSE-formatted strings so
that BedrockAgentCoreApp serves /invocations with Content-Type:
text/event-stream.  The frontend reads the response body as a ReadableStream
and renders tokens incrementally.

SSE event schema (public, stable across versions)
--------------------------------------------------
Each line on the wire is:  data: <JSON>\n\n

  {"type": "text",  "text": "<token>"}
      A piece of the model response.  The frontend appends it.

  {"type": "tool",  "name": "<tool_name>", "status": "started"|"done"}
      A tool call started or finished.  The frontend may show a spinner.

  {"type": "done",  "result": "<full_response>"}
      The run is complete.  The frontend replaces the streamed text with
      this authoritative value and closes the stream.

  {"type": "error", "error": "<message>"}
      A fatal error.  The frontend shows the message and stops reading.

Only these four types are emitted.  Every other Strands event is dropped.

Strands 1.65.0 stream_async event mapping
------------------------------------------
  TextStreamEvent       → {"data": str, "delta": {...}}
  ToolUseStreamEvent    → {"type": "tool_use_stream", "current_tool_use": {...}, "delta": {...}}
  ToolResultMessageEvent→ {"message": {...}}
  AgentResultEvent      → {"result": AgentResult}  (str(result) = full text)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig,
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager,
)
from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ============================================================
# CONFIGURACIÓN
# ============================================================

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID: str = "us.amazon.nova-2-lite-v1:0"

MEMORY_ID: str | None = os.environ.get("MEMORY_ID")
if not MEMORY_ID:
    raise EnvironmentError(
        "MEMORY_ID environment variable is required but not set."
    )

_MCP_URL: str | None = os.environ.get("MCP_URL")
if not _MCP_URL:
    raise EnvironmentError(
        "MCP_URL environment variable is required but not set. "
        "Set it to the full URL of the MCP server endpoint, "
        "e.g. https://<agentcore-endpoint>/mcp"
    )
MCP_URL: str = _MCP_URL


# ============================================================
# SYSTEM PROMPT  (copied verbatim from agent.py — do not alter)
# ============================================================

SYSTEM_PROMPT = """
You are an optimization assistant specialized in classical and
hybrid optimization using QAOA.

Your role is to help users formulate selection problems, solve
them classically, explore QAOA simulations, estimate resources,
and refine QAOA solutions using classical methods.

AVAILABLE TOOLS

1. formulate_problem
   Registers a selection problem.
   Requires the user's items, values, costs, and budget.
   Returns problem_id.

2. solve_classical
   Solves a registered problem using either:
   - exact: exhaustive search
   - greedy: heuristic optimization
   Returns a classical run_id.
   This tool works independently of QAOA.

3. run_qaoa
   Executes QAOA using the local Braket simulator.
   Requires problem_id.
   Returns a QAOA run_id.
   This is a classical simulation, not a QPU execution.

4. hybrid_shortlist
   Refines candidates from a successful QAOA execution
   using classical repair and optional polishing.
   Requires qaoa_run_id returned by run_qaoa.
   Returns a shortlist run_id.

5. analyze_results
   Analyzes QAOA and/or hybrid results.
   Requires problem_id and at least one of:
   - qaoa_run_id
   - shortlist_run_id
   Both IDs must belong to the same problem.
   Never pass a classical run_id.

6. estimate_resources
   Estimates resources for a registered problem.
   Requires problem_id.
   Does not execute QAOA or create a Braket task.
   Can be used independently of run_qaoa.

ORCHESTRATION RULES

- First identify the user's actual intent.
- Answer conceptual questions directly without calling tools.
- Do not formulate a problem unless the user requests an
  operation that requires a registered problem.
- If the user provides all required problem data, call
  formulate_problem once without asking for confirmation.
- Never invent items, values, costs, budgets, IDs, or results.
- If required data is missing, ask for it and offer a concrete
  example. If the user accepts the example, use it explicitly.

Choose only the workflow requested:

- Classical solution:
  formulate_problem -> solve_classical

- QAOA experiment:
  formulate_problem -> run_qaoa

- Hybrid refinement:
  formulate_problem -> run_qaoa -> hybrid_shortlist

- Resource estimation:
  formulate_problem -> estimate_resources

- QAOA or hybrid analysis:
  use analyze_results with valid IDs from previous executions.

- Classical and QAOA comparison:
  obtain the classical and QAOA results independently.
  Never pass the classical run_id to analyze_results.
  Explain the comparison using the actual returned results.

- Do not automatically run classical solving, QAOA,
  hybrid refinement, or analysis unless requested or
  explicitly required by the selected workflow.

STATE AND EXECUTION SAFETY

- Preserve exact problem_id and run_id values returned by tools.
- Reuse IDs only when their associated records are still valid.
- Retrieve IDs from conversation context when available.
- If a required ID cannot be recovered, ask the user.
- Do not repeat successful executions unnecessarily.
- Never claim success unless the tool confirms it.
- Omit unused optional arguments instead of sending null
  or empty values.

QUANTUM COMPUTING ACCURACY

- The local Braket simulator is classical simulation,
  not a real quantum processor.
- Never claim quantum advantage from local simulation.
- Distinguish QAOA exploration from classical refinement.
- Do not describe simulated results as hardware measurements.
- Report only metrics actually returned by tools.
- Explain relevant limitations when comparing approaches.

RESPONSE STYLE

- Respond in the user's language.
- Be clear, concise, and technically accurate.
- Explain which tools and optimization approaches were used.
- Summarize actual results and relevant limitations.
- Do not expose internal reasoning.
- If the user sends an empty message, do nothing.
"""


# ============================================================
# AGENTCORE APP
# ============================================================

app = BedrockAgentCoreApp()


# ============================================================
# SHARED RESOURCES  (initialized once, reused across invocations)
# ============================================================

_EXPOSED_TOOLS: frozenset[str] = frozenset({
    "formulate_problem",
    "solve_classical",
    "run_qaoa",
    "hybrid_shortlist",
    "analyze_results",
    "estimate_resources",
})

# Lazy-initialization state protected by a lock so that concurrent
# first-calls (AgentCore may use multiple threads) only run setup once.
_init_lock: threading.Lock = threading.Lock()
_initialized: bool = False

# These are populated by _ensure_initialized() and are safe to read
# from any thread once _initialized is True.
_model: BedrockModel | None = None
_mcp_client: MCPClient | None = None
_agent_tools: list | None = None   # filtered, validated list of MCP tools


def _mcp_transport_factory():
    """Return the SigV4-signed Streamable HTTP transport for MCPClient."""
    return aws_iam_streamablehttp_client(
        endpoint=MCP_URL,
        aws_service="bedrock-agentcore",
        aws_region=AWS_REGION,
    )


def _ensure_initialized() -> None:
    """Lazily initialize shared resources, at most once per process.

    The double-checked locking pattern avoids acquiring the lock on the
    hot path (every call after the first).

    Raises
    ------
    RuntimeError
        If the MCP server does not expose all tools in _EXPOSED_TOOLS.
    """
    global _initialized, _model, _mcp_client, _agent_tools

    if _initialized:
        return  # fast path — no lock needed after first successful init

    with _init_lock:
        if _initialized:
            return  # another thread finished while we waited for the lock

        logger.info("Initializing shared resources (first invocation)…")
        t0 = time.perf_counter()

        # --- BedrockModel ------------------------------------------------
        session = boto3.Session(region_name=AWS_REGION)
        _model = BedrockModel(
            boto_session=session,
            model_id=MODEL_ID,
            temperature=0,
        )
        logger.info("BedrockModel ready (%.3fs)", time.perf_counter() - t0)

        # --- MCPClient: open the connection and keep it alive ------------
        # MCPClient normally expects a `with` block.  We enter the context
        # once here and never exit it, so the connection stays open for
        # the entire process lifetime.  The process ends when AgentCore
        # tears down the runtime, at which point the OS cleans up the
        # underlying socket.
        _mcp_client = MCPClient(
            _mcp_transport_factory,
            startup_timeout=30,
        )
        _mcp_client.__enter__()  # opens the transport and negotiates MCP
        logger.info("MCP connection open (%.3fs)", time.perf_counter() - t0)

        # --- Discover and validate tools ---------------------------------
        all_tools = _mcp_client.list_tools_sync()
        _agent_tools = [
            tool for tool in all_tools
            if tool.tool_name in _EXPOSED_TOOLS
        ]

        loaded = {tool.tool_name for tool in _agent_tools}
        missing = _EXPOSED_TOOLS - loaded
        if missing:
            # Clean up before raising so the process doesn't hold a
            # half-initialized MCP connection.
            try:
                _mcp_client.__exit__(None, None, None)
            except Exception:
                pass
            _mcp_client = None
            _agent_tools = None
            raise RuntimeError(
                f"Required MCP tools are unavailable: {sorted(missing)}"
            )

        logger.info(
            "Tools ready — %d exposed tools loaded (%.3fs)",
            len(_agent_tools),
            time.perf_counter() - t0,
        )

        _initialized = True
        logger.info(
            "Shared resource initialization complete (%.3fs)",
            time.perf_counter() - t0,
        )


# ============================================================
# SSE HELPERS
# ============================================================

def _sse(payload: dict) -> str:
    """Format a dict as a single SSE data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ============================================================
# ENTRYPOINT
# ============================================================

@app.entrypoint
async def handler(payload: dict):
    """AgentCore streaming invocation handler.

    Declared as an async generator: each ``yield`` sends one SSE frame.
    BedrockAgentCoreApp detects the generator and sets
    Content-Type: text/event-stream automatically.

    Only per-request work happens here: input validation, session-manager
    construction, and the agent call.  All heavy initialization is done
    once by _ensure_initialized().

    SSE events yielded
    ------------------
    {"type": "text",  "text": "<token>"}
    {"type": "tool",  "name": "<name>", "status": "started"|"done"}
    {"type": "done",  "result": "<full_response>"}
    {"type": "error", "error": "<message>"}

    Parameters
    ----------
    payload : dict
        Must contain non-empty string fields ``prompt``, ``actor_id``,
        and ``session_id``.
    """
    t_enter = time.perf_counter()
    logger.info("[handler] enter")

    # --- Input validation ------------------------------------------------
    prompt: str | None = payload.get("prompt")
    if not prompt or not isinstance(prompt, str) or not prompt.strip():
        yield _sse({"type": "error", "error": "payload must contain a non-empty 'prompt' string field."})
        return

    actor_id = payload.get("actor_id")
    session_id = payload.get("session_id")
    if not isinstance(actor_id, str) or not actor_id.strip():
        yield _sse({"type": "error", "error": "payload must contain a non-empty 'actor_id' string field."})
        return
    if not isinstance(session_id, str) or not session_id.strip():
        yield _sse({"type": "error", "error": "payload must contain a non-empty 'session_id' string field."})
        return

    # --- Ensure shared resources are ready (no-op after first call) ------
    try:
        _ensure_initialized()
    except Exception as exc:
        logger.exception("[handler] initialization failed")
        yield _sse({"type": "error", "error": str(exc)})
        return

    t_tools = time.perf_counter()
    logger.info(
        "[handler] tools ready (%.3fs since enter — expect ~0 after warmup)",
        t_tools - t_enter,
    )

    # --- Per-request session manager -------------------------------------
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        actor_id=actor_id.strip(),
        session_id=session_id.strip(),
    )
    session_manager = AgentCoreMemorySessionManager(
        agentcore_memory_config=memory_config,
        region_name=AWS_REGION,
    )

    t_session = time.perf_counter()
    logger.info("[handler] session manager built (%.3fs since enter)", t_session - t_enter)

    # --- Agent call (streaming) ------------------------------------------
    # Agent is reconstructed per request because session_manager carries
    # per-session state.  BedrockModel and tools are reused from the
    # module-level singletons.
    #
    # callback_handler=None silences the default stdout printer so tokens
    # don't get double-printed to logs.
    #
    # Event mapping for Strands 1.65.0 stream_async:
    #   Text token  → {"data": str, "delta": {...}}                   → SSE type:text
    #   Tool stream → {"type": "tool_use_stream", "current_tool_use"} → SSE type:tool started
    #   Final       → {"result": AgentResult}                         → SSE type:done (also closes tools)
    agent = Agent(
        model=_model,
        tools=_agent_tools,
        system_prompt=SYSTEM_PROMPT,
        session_manager=session_manager,
        callback_handler=None,
    )

    full_text: list[str] = []
    # Track active tool calls by toolUseId so we can emit "done" when they finish.
    _active_tools: dict[str, str] = {}  # toolUseId -> tool_name

    try:
        async for event in agent.stream_async(prompt.strip()):
            if not isinstance(event, dict):
                continue

            # ── text delta ───────────────────────────────────────────────
            # Strands 1.65.0: TextStreamEvent yields {"data": "<text>", "delta": {...}}
            # Reasoning events also have "delta" but carry "reasoning": True — skip those.
            if (
                "data" in event
                and "delta" in event
                and not event.get("reasoning")
                and isinstance(event["data"], str)
            ):
                text_chunk = event["data"]
                if text_chunk:
                    full_text.append(text_chunk)
                    yield _sse({"type": "text", "text": text_chunk})
                continue

            # ── tool started ─────────────────────────────────────────────
            # Strands 1.65.0: ToolUseStreamEvent yields
            # {"type": "tool_use_stream", "delta": {...}, "current_tool_use": {"name": ..., "toolUseId": ...}}
            if event.get("type") == "tool_use_stream":
                current = event.get("current_tool_use", {})
                tool_name = current.get("name", "tool")
                tool_id = current.get("toolUseId", tool_name)
                if tool_id not in _active_tools:
                    _active_tools[tool_id] = tool_name
                    yield _sse({"type": "tool", "name": tool_name, "status": "started"})
                continue

            # ── tool finished ────────────────────────────────────────────
            # ── final result ─────────────────────────────────────────────
            # AgentResultEvent yields {"result": AgentResult}.
            # We close any still-open tool indicators here before the done event.
            # str(result) concatenates text blocks from result.message["content"].
            if "result" in event:
                # Close any tool calls that were started but not yet closed.
                for tool_id, tool_name in list(_active_tools.items()):
                    yield _sse({"type": "tool", "name": tool_name, "status": "done"})
                _active_tools.clear()
                agent_result = event["result"]
                result_text = str(agent_result).strip()
                yield _sse({"type": "done", "result": result_text})
                t_exit = time.perf_counter()
                logger.info("[handler] done — total %.3fs", t_exit - t_enter)
                return

            # All other events (lifecycle, reasoning, metrics) are dropped.

    except Exception as exc:
        logger.exception("[handler] stream_async failed")
        yield _sse({"type": "error", "error": str(exc)})
        return

    # --- Fallback final event --------------------------------------------
    # Reached only if AgentResultEvent was never emitted (should not happen
    # in normal operation, but guards against truncated streams).
    result = "".join(full_text)
    yield _sse({"type": "done", "result": result})
    logger.info("[handler] done (fallback) — total %.3fs", time.perf_counter() - t_enter)


# ============================================================
# LOCAL EXECUTION
# ============================================================

if __name__ == "__main__":
    app.run()

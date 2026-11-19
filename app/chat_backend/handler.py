"""
POST /chat — FastAPI streaming handler for Qonnect.

Runs as a FastAPI app inside Lambda via Lambda Web Adapter (LWA).
LWA starts uvicorn and forwards HTTP requests to FastAPI. The response
body is streamed progressively through the Lambda Function URL.

Flow
----
Browser ──POST /chat──▶ Lambda Function URL  (InvokeMode: RESPONSE_STREAM)
                       ▶ LWA extension       (port 8080)
                       ▶ FastAPI / uvicorn   (this file)
                       ▶ boto3 InvokeAgentRuntime  →  AgentCore SSE stream
                       ▶ progressively yields received SSE chunks

Timestamps logged to CloudWatch (time.perf_counter())
-----------------------------------------------------
t0  FastAPI route entered
t1  invoke_agent_runtime dispatched
t2  first non-empty SSE chunk received from AgentCore
t3  stream generator completed

These checkpoints are logged to CloudWatch. They are not all HTTP headers.

Response headers
----------------
X-Qonnect-T0        perf_counter timestamp at handler entry
X-Qonnect-Parse-Ms  time from handler entry until request validation
                    and response startup

The t1, t2 and t3 measurements are emitted as TIMING log entries.
Streaming behavior is progressive, but buffering may occur in components
outside this handler. Per-token delivery is not guaranteed.

Environment variables
---------------------
AGENT_RUNTIME_ARN  ARN of the AgentCore Runtime (quantum_agent)
AWS_REGION         Region — set by Lambda automatically
CORS_ORIGIN        Access-Control-Allow-Origin (default: *)
PORT               Port uvicorn listens on — set by LWA (default 8080)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncGenerator

import boto3
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("qonnect.chat")

# ── environment ────────────────────────────────────────────────────────────────

AGENT_RUNTIME_ARN: str = os.environ["AGENT_RUNTIME_ARN"]
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
CORS_ORIGIN: str = os.environ.get("CORS_ORIGIN", "*")

# boto3 client at module level — reused across warm invocations (~50 ms saved
# per request on the connection-setup path).
_agentcore = boto3.client("bedrock-agentcore", region_name=AWS_REGION)

# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title="Qonnect chat backend", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Amzn-Trace-Id"],
    max_age=3600,
)


# ── streaming helpers ──────────────────────────────────────────────────────────

def _invoke_and_stream(
    session_id: str,
    invoke_payload: bytes,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
) -> None:
    """Read AgentCore's synchronous stream and forward each line to asyncio."""
    response = None

    def publish(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        response = _agentcore.invoke_agent_runtime(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeSessionId=session_id,
            payload=invoke_payload,
        )

        content_type = response.get("contentType", "")
        publish(("content_type", content_type))

        if "text/event-stream" in content_type:
            for raw_line in response["response"].iter_lines(chunk_size=1):
                if raw_line:
                    line = (
                        raw_line.decode("utf-8")
                        if isinstance(raw_line, bytes)
                        else raw_line
                    )
                    publish(("line", line))
        else:
            raw_chunks = []
            for chunk in response.get("response", []):
                raw_chunks.append(
                    chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                )
            try:
                parsed = json.loads("".join(raw_chunks)) if raw_chunks else {}
                result_text = parsed.get("result", "")
            except Exception:
                result_text = ""
            publish((
                "line",
                f'data: {json.dumps({"type": "done", "result": result_text})}',
            ))

    except Exception as exc:
        logger.exception("invoke_agent_runtime failed")
        publish(("error", "No se pudo completar la solicitud. Intenta de nuevo."))
    finally:
        if response is not None:
            try:
                response["response"].close()
            except Exception:
                pass
        publish(("end", None))


async def _stream_agent(
    prompt: str,
    actor_id: str,
    session_id: str,
    t0: float,
) -> AsyncGenerator[bytes, None]:
    """Forward AgentCore SSE events to the browser as soon as they arrive."""
    invoke_payload = json.dumps({
        "prompt": prompt,
        "actor_id": actor_id,
        "session_id": session_id,
    }).encode()

    t1 = time.perf_counter()
    logger.info(
        "TIMING t1=%.6f dt=%.3fs | invoke_agent_runtime dispatched session=%s",
        t1, t1 - t0, session_id,
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    worker = loop.run_in_executor(
        None,
        _invoke_and_stream,
        session_id,
        invoke_payload,
        loop,
        queue,
    )

    first_chunk = True
    try:
        while True:
            kind, value = await queue.get()

            if kind == "end":
                break

            if kind == "error":
                yield f'data: {json.dumps({"type": "error", "error": value})}\n\n'.encode()
                break

            if kind == "content_type":
                logger.info("AgentCore contentType=%s", value)
                continue

            if kind == "line":
                if first_chunk:
                    t2 = time.perf_counter()
                    logger.info(
                        "TIMING t2=%.6f dt_since_t0=%.3fs dt_since_invoke=%.3fs | first chunk received",
                        t2, t2 - t0, t2 - t1,
                    )
                    first_chunk = False

                frame = value if value.endswith("\n\n") else value.rstrip("\n") + "\n\n"
                yield frame.encode()

    finally:
        await worker

    t3 = time.perf_counter()
    logger.info(
        "TIMING t3=%.6f dt_total=%.3fs | generator done",
        t3, t3 - t0,
    )


# ── routes ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    """Stream SSE tokens from AgentCore back to the browser."""
    t0 = time.perf_counter()
    logger.info("TIMING t0=%.6f wall=%.6f | handler entered", t0, time.time())

    # ── parse body ────────────────────────────────────────────────────────────
    try:
        body: dict = await request.json()
    except Exception:
        return Response(
            content=json.dumps({"error": "Request body must be valid JSON."}),
            status_code=400,
            media_type="application/json",
        )

    prompt = body.get("prompt")
    actor_id = body.get("actor_id")
    session_id = body.get("session_id")

    if not isinstance(prompt, str) or not prompt.strip():
        return Response(
            content=json.dumps({"error": "Field 'prompt' must be a non-empty string."}),
            status_code=400,
            media_type="application/json",
        )
    if not isinstance(actor_id, str) or not actor_id.strip():
        return Response(
            content=json.dumps({"error": "Field 'actor_id' must be a non-empty string."}),
            status_code=400,
            media_type="application/json",
        )
    if not isinstance(session_id, str) or not session_id.strip():
        return Response(
            content=json.dumps({"error": "Field 'session_id' must be a non-empty string."}),
            status_code=400,
            media_type="application/json",
        )

    t1_header = f"{(time.perf_counter() - t0) * 1000:.1f}"  # ms to validate+start

    return StreamingResponse(
        _stream_agent(prompt.strip(), actor_id.strip(), session_id.strip(), t0),
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            # t0 and the parse cost are knowable at response-creation time.
            # t2/t3 are logged to CloudWatch (they happen inside the generator
            # after the HTTP header section is already sent to the client).
            "X-Qonnect-T0": f"{t0:.6f}",
            "X-Qonnect-Parse-Ms": t1_header,
        },
    )


@app.get("/health")
async def health() -> dict:
    """LWA readiness probe and manual smoke-test endpoint."""
    return {"status": "ok"}

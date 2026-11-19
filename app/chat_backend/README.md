# chat_backend

FastAPI backend that exposes `POST /chat` through an AWS Lambda Function URL
and forwards AgentCore Server-Sent Events (SSE) progressively to the browser.

## Architecture

    Browser (web/)
      | POST /chat
      | fetch + ReadableStream
      v
    Lambda Function URL (InvokeMode: RESPONSE_STREAM)
      |
      v
    Lambda Web Adapter (LWA)
      |
      v
    FastAPI / uvicorn (handler.py, port 8080)
      |
      v
    boto3 bedrock-agentcore.invoke_agent_runtime
      | synchronous response stream, consumed in a worker thread
      v
    AgentCore Runtime: quantum_agent

The backend forwards non-empty SSE lines as they arrive from AgentCore.
FastAPI uses `StreamingResponse`, and the browser reads the response
progressively.

**Validation status:** progressive streaming has been tested with a simulated
AgentCore response. The complete path from a real AgentCore invocation through
Lambda and the browser has not yet been validated end to end. Do not interpret
this as a guarantee of per-token delivery; buffering may occur in components
outside this handler.

## Streaming in Python

The Lambda uses AWS Lambda Web Adapter (LWA) to run FastAPI with response
streaming.

Relevant configuration:

- `AWS_LAMBDA_EXEC_WRAPPER=/opt/bootstrap`
- `AWS_LWA_INVOKE_MODE=response_stream`
- `PORT=8080`
- `run.sh` starts uvicorn and loads `handler:app`.

The Lambda Function URL must use `InvokeMode.RESPONSE_STREAM`.

## Timing instrumentation

The handler writes four timing checkpoints to CloudWatch Logs:

| Checkpoint | Meaning |
|---|---|
| `t0` | FastAPI route entered |
| `t1` | `invoke_agent_runtime` dispatched |
| `t2` | First non-empty SSE chunk received from AgentCore |
| `t3` | Stream generator completed |

These checkpoints are log entries, not four HTTP response headers.

The endpoint currently returns these measurement headers:

| Header | Meaning |
|---|---|
| `X-Qonnect-T0` | `perf_counter` timestamp at handler entry |
| `X-Qonnect-Parse-Ms` | Time from handler entry until request validation and response startup |

The `t1`, `t2` and `t3` measurements are logged in CloudWatch. They are not
returned as response headers.

**CloudWatch Logs Insights:**

    fields @timestamp, @message
    | filter @message like "TIMING"
    | sort @timestamp desc
    | limit 100

No cold-start or warm-start latency figures are documented because they have
not yet been measured in the deployed environment.

## Packaging for Lambda

The CDK asset uses `app/chat_backend/build`, not the source directory directly.
Generate this directory before synthesizing or deploying the CDK stack.

From the repository root:

    ./app/chat_backend/package.sh

The script:

- Exports locked runtime dependencies from `uv.lock`.
- Installs binary dependencies for Python 3.13 and Linux ARM64.
- Copies `handler.py` and `run.sh` into `build/`.
- Produces the directory consumed by CDK.

The generated `build/` directory is ignored by Git. Run the packaging script
again whenever the backend dependencies or source files change.

## Deployment prerequisites

Deployment is intentionally not performed as part of local development.

Before deploying:

1. Ensure the `quantum_agent` AgentCore Runtime is deployed and available.
2. Obtain its actual runtime ARN from the AWS environment.
3. Ensure the CDK deployment targets the same AWS region as that runtime.
4. Package the Lambda from the repository root:

       ./app/chat_backend/package.sh

5. From `agentcore/cdk/`, provide the runtime ARN through either the
   `AGENT_RUNTIME_ARN` environment variable or the CDK context key
   `agentRuntimeArn`.

   Example using an environment variable (replace the placeholder locally;
   do not commit the ARN):

       AGENT_RUNTIME_ARN="<actual-runtime-arn>" npx cdk deploy

   Alternatively, use CDK context:

       npx cdk deploy --context agentRuntimeArn="<actual-runtime-arn>"

The CDK stack intentionally fails if the runtime ARN is missing. It does not
use a placeholder ARN.

After an authorized deployment, CDK outputs `ChatFunctionUrl`. Configure
`VITE_API_URL` in `web/.env.local` with the Function URL base. The frontend
sends requests to the `/chat` endpoint.

## Security considerations

The current Function URL is configured with `AuthType.NONE` and CORS allows
all origins (`*`). This is a public endpoint and is suitable only for a
controlled demo or MVP.

Before production use, add an appropriate authentication and abuse-protection
strategy, restrict CORS to the actual frontend origin, and review IAM,
logging, quotas and cost exposure.

## Local development

Use Python 3.13 for the backend environment.

    cd app/chat_backend
    uv sync

Run the server using an actual deployed AgentCore runtime ARN and your
configured AWS credentials:

    AGENT_RUNTIME_ARN="<actual-runtime-arn>" uv run uvicorn handler:app --reload --host 0.0.0.0 --port 8080

Set the frontend API base in `web/.env.local`:

    VITE_API_URL=http://localhost:8080/

Then start the frontend from `web/`:

    npm run dev

The `/health` endpoint can be used for a basic local availability check.

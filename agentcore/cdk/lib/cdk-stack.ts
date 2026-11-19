import * as path from 'path';
import { AgentCoreApplication, type AgentCoreApplicationProps } from '@aws/agentcore-cdk';
import {
  CfnOutput,
  Duration,
  Stack,
  type StackProps,
  aws_iam as iam,
  aws_lambda as lambda,
  aws_logs as logs,
} from 'aws-cdk-lib';
import { Construct } from 'constructs';

export interface AgentCoreStackProps extends StackProps {
  /** Props for the AgentCore application, produced by transformAgentCoreJson from agentcore/agentcore.json. */
  application: AgentCoreApplicationProps;
}

/**
 * AgentCore stack.
 *
 * Deploys the AgentCore application declared in agentcore/agentcore.json
 * and a /chat Lambda that forwards browser requests to the quantum_agent
 * AgentCore Runtime.
 *
 * Streaming architecture
 * ----------------------
 * The chat backend uses Lambda Web Adapter (LWA) to run FastAPI with
 * uvicorn and Lambda Function URL response streaming:
 *
 *   Browser ──POST /chat──▶ Lambda Function URL (RESPONSE_STREAM)
 *                          ▶ LWA extension
 *                          ▶ FastAPI/uvicorn (handler.py)
 *                          ▶ boto3 InvokeAgentRuntime (SSE response)
 *                          ▶ progressively forwards received SSE chunks
 *
 * This configuration enables response streaming at the Lambda Function URL.
 * End-to-end delivery timing and buffering depend on the components in the
 * request path; per-token delivery is not guaranteed.
 *
 * VITE_API_URL after deployment
 * -----------------------------
 * Set web/.env.local to the ChatFunctionUrl output value. The frontend
 * sends requests to the /chat route handled by FastAPI.
 *
 * Cold-start diagnosis
 * --------------------
 * X-Ray tracing is enabled. Compare initialization and invocation telemetry
 * in the deployed environment to measure cold-start and warm-start behavior.
 */
export class AgentCoreStack extends Stack {
  public readonly application: AgentCoreApplication;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);
    this.application = new AgentCoreApplication(this, 'Application', props.application);

    // ── Resolve the quantum_agent runtime ARN ─────────────────────────────
    // Supply the deployed runtime ARN through CDK context or the environment.
    // The stack fails explicitly when neither source provides an ARN.
    const agentRuntimeArn =
      (this.node.tryGetContext('agentRuntimeArn') as string | undefined)
      ?? process.env['AGENT_RUNTIME_ARN'];

    if (!agentRuntimeArn) {
      throw new Error(
        'Missing AgentCore runtime ARN. Set AGENT_RUNTIME_ARN or CDK context agentRuntimeArn.',
      );
    }

    // ── Lambda execution role ─────────────────────────────────────────────
    const chatRole = new iam.Role(this, 'ChatLambdaRole', {
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaBasicExecutionRole',
        ),
      ],
    });

    chatRole.addToPolicy(new iam.PolicyStatement({
      actions: ['bedrock-agentcore:InvokeAgentRuntime'],
      // Wildcard suffix covers qualifier-suffixed ARNs (:live, :version, etc.)
      resources: [`${agentRuntimeArn}*`],
    }));

    chatRole.addToPolicy(new iam.PolicyStatement({
      actions: ['xray:PutTraceSegments', 'xray:PutTelemetryRecords'],
      resources: ['*'],
    }));

    // ── Lambda Web Adapter layer ──────────────────────────────────────────
    // Public AWS Lambda Web Adapter layer for the ARM64 architecture.
    // Review the published layer version when updating the deployment.
    const lwaLayer = lambda.LayerVersion.fromLayerVersionArn(
      this,
      'LambdaWebAdapterLayer',
      `arn:aws:lambda:${this.region}:753240598075:layer:LambdaAdapterLayerArm64:30`,
    );

    // ── Lambda function ───────────────────────────────────────────────────
    const chatLambda = new lambda.Function(this, 'ChatLambda', {
      functionName: `${this.stackName}-chat`,
      runtime: lambda.Runtime.PYTHON_3_13,
      architecture: lambda.Architecture.ARM_64,
      // handler is the LWA startup script run.sh, not a Python module.
      // LWA's wrapper (/opt/bootstrap) intercepts the execution and runs run.sh,
      // which starts uvicorn.  handler.py is the FastAPI app uvicorn loads.
      handler: 'run.sh',
      code: lambda.Code.fromAsset(
        path.join(__dirname, '..', '..', '..', 'app', 'chat_backend', 'build'),
      ),
      layers: [lwaLayer],
      memorySize: 256,
      // Maximum configured Lambda execution duration (15 minutes).
      // AgentCore execution may still have its own duration limits.
      timeout: Duration.seconds(900),
      role: chatRole,
      environment: {
        // LWA environment variables
        AWS_LAMBDA_EXEC_WRAPPER: '/opt/bootstrap',   // LWA wrapper script
        AWS_LWA_INVOKE_MODE: 'response_stream',       // tells LWA to stream
        PORT: '8080',                                 // uvicorn port

        // Application environment variables
        AGENT_RUNTIME_ARN: agentRuntimeArn,
        // CORS_ORIGIN: 'https://your-cloudfront-domain.cloudfront.net',
      },
      tracing: lambda.Tracing.ACTIVE,
      logRetention: logs.RetentionDays.ONE_WEEK,
    });

    // ── Optional Provisioned Concurrency ──────────────────────────────────
    // Evaluate its latency benefit and ongoing cost before enabling.
    // const chatVersion = chatLambda.currentVersion;
    // new lambda.Alias(this, 'ChatLambdaLive', {
    //   aliasName: 'live',
    //   version: chatVersion,
    //   provisionedConcurrentExecutions: 1,
    // });

    // ── Function URL (RESPONSE_STREAM) ────────────────────────────────────
    // Enables Lambda's response-streaming invocation mode. Progressive
    // end-to-end delivery also depends on the behavior of upstream components.
    const chatUrl = chatLambda.addFunctionUrl({
      authType: lambda.FunctionUrlAuthType.NONE,
      invokeMode: lambda.InvokeMode.RESPONSE_STREAM,
      cors: {
        allowedOrigins: ['*'],          // restrict in production
        allowedHeaders: ['Content-Type', 'X-Amzn-Trace-Id'],
        allowedMethods: [lambda.HttpMethod.POST],
        maxAge: Duration.hours(1),
      },
    });

    // ── Outputs ───────────────────────────────────────────────────────────
    new CfnOutput(this, 'ChatFunctionUrl', {
      description:
        'Base URL for the /chat endpoint — use as VITE_API_URL in web/.env.local',
      value: chatUrl.url,
      exportName: `${this.stackName}-ChatFunctionUrl`,
    });

    new CfnOutput(this, 'ChatLambdaArn', {
      description: 'ARN of the /chat Lambda (CloudWatch Logs, X-Ray)',
      value: chatLambda.functionArn,
    });
  }
}

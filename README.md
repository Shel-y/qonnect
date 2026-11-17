# Qonnect

### Connecting AI Agents with Quantum Optimization through MCP

**Created by Joselyn Lagunas**

Qonnect is an experimental project exploring how AI agents can interact with quantum computing and hybrid optimization capabilities through the **Model Context Protocol (MCP)**.

At its core, Qonnect provides an MCP server designed to expose optimization capabilities as tools that AI agents can discover and use. The goal is to make quantum and classical optimization workflows more accessible through standardized agent-tool interactions.

## Why MCP?

The Model Context Protocol provides a standardized way for AI applications to communicate with external tools and services.

In Qonnect, MCP acts as the integration layer between AI agents and optimization capabilities. Rather than embedding optimization logic directly into an agent, the MCP server exposes these capabilities through a reusable tool interface.

This architecture explores:

- Tool-based access to optimization workflows.
- Integration between AI agents and quantum computing services.
- Hybrid approaches combining classical and quantum computing.
- Reusable interfaces for experimentation and future extensions.

## Architecture

Qonnect is organized around three main components:

1. **Qonnect MCP Server** — The core component. Exposes optimization capabilities through MCP tools and runs on Amazon Bedrock AgentCore with IAM authentication.
2. **Quantum Agent** — A Strands-based AI agent that interacts with MCP tools and provides a conversational layer.
3. **Web Interface** — A React and TypeScript application for interacting with the project.

The MCP server is the central integration layer, while the agent and frontend provide complementary ways to access its capabilities.

## Technology Stack

- Model Context Protocol (MCP)
- Python
- Amazon Bedrock AgentCore
- Strands Agents
- AWS
- React
- TypeScript
- Vite
- Classical and quantum optimization experimentation

## Repository Structure

```text
qonnect/
├── agentcore/
│   └── cdk/
├── app/
│   ├── quantum_mcp/
│   └── quantum_agent/
└── web/
```

## Project Status

Qonnect is an experimental project under active development.

The MCP server and AI agent are deployed independently on Amazon Bedrock AgentCore. The web interface is being prepared for integration with the deployed agent through a secure backend.

The project explores hybrid quantum-classical optimization as an emerging capability, not as a replacement for classical computing.

## Vision

Make quantum computing capabilities easier to explore, integrate, and use through AI agents and open, standardized tool interfaces.

**Qonnect: connecting agents, tools, and quantum possibilities.**

## Author

**Joselyn Lagunas**

Emerging Technologies & Innovation Advocate | AWS Community Builder

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
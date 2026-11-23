# ⚛️ Qonnect

<h3 align="center">Connecting AI Agents with Classical & Quantum Optimization through MCP\</h3>

<p align="center">
&#x20; An experimental platform exploring how AI agents can orchestrate classical and quantum optimization tools through the Model Context Protocol.
</p>

<p align="center">
&#x20; \<img src="[https://img.shields.io/badge/Python-3.13-blue?logo=python](https://img.shields.io/badge/Python-3.13-blue?logo=python)" alt="Python 3.13">
&#x20; \<img src="[https://img.shields.io/badge/AI-Strands%20Agents-purple](https://img.shields.io/badge/AI-Strands%20Agents-purple)" alt="Strands Agents">
&#x20; \<img src="[https://img.shields.io/badge/AWS-Amazon%20Braket-orange?logo=amazonaws](https://img.shields.io/badge/AWS-Amazon%20Braket-orange?logo=amazonaws)" alt="Amazon Braket">
&#x20; \<img src="[https://img.shields.io/badge/Protocol-MCP-8A2BE2](https://img.shields.io/badge/Protocol-MCP-8A2BE2)" alt="MCP">
&#x20; \<img src="[https://img.shields.io/badge/Status-Experimental-yellow](https://img.shields.io/badge/Status-Experimental-yellow)" alt="Experimental"></p>

<p align="center">
&#x20; <strong>Created by <a href="[https://www.linkedin.com/in/joselyn-lagunas/](https://www.linkedin.com/in/joselyn-lagunas/)">Joselyn Lagunas</a></strong>
</p>

---

## ✨ See Qonnect in action

<!-- Replace with your actual demo GIF -->

<div align="center">
  <img
    src="https://github.com/Shel-y/qonnect/blob/f47611ab54b473818204c6a10a354d2769af28e6/assets/qonnect-demo.gif"
    width="100%"
    alt="Qonnect Demo"
  />
</div>

<p align="center">
&#x20; <em>From natural language requests to coordinated optimization workflows.</em>
</p>

---

## 💡 What is Qonnect?

What if an AI agent could access quantum computing tools to experiment with optimization problems?

**Qonnect explores this idea.**

It connects an AI agent to a dedicated MCP server that exposes classical and quantum optimization capabilities. Through a conversational interface, users can describe a problem, request a solution, estimate resources, and analyze results.

The goal is to make emerging optimization technologies accessible through agentic AI.

## 🏗️ Architecture

<p align="center">
&#x20; <img src="assets/qonnect-architecture.svg" alt="Qonnect architecture diagram" width="100%">
</p>

### How it works

```mermaid
flowchart TD
    A[" React Frontend<br/>AWS Amplify"]
    B["AWS Lambda Function URL"]
    C[" AWS Lambda<br/>FastAPI + Python"]
    D[" quantum_agent<br/>Strands Agents + Nova 2 Lite"]
    E[" quantum_mcp<br/>AgentCore Runtime"]
    F["Classical Optimization"]
    G["Amazon Braket<br/>QAOA"]

    A --> B --> C --> D --> E
    E --> F
    E --> G

    C -. "SSE response stream" .-> A

    H["AgentCore Memory"]
    D <--> H
```

### Architecture components

| Component           | Responsibility                                             |
| ------------------- | ---------------------------------------------------------- |
| **Frontend**        | React + TypeScript conversational interface                |
| **API Entry Point** | AWS Lambda Function URL                                    |
| **Backend**         | FastAPI running on AWS Lambda                              |
| **AI Agent**        | Strands Agents on Amazon Bedrock AgentCore Runtime         |
| **MCP Server**      | Independent AgentCore Runtime exposing optimization tools  |
| **Optimization**    | Classical solving and experimental QAOA with Amazon Braket |
| **Memory**          | AgentCore Memory for contextual continuity                 |

The agent and MCP server run in separate AgentCore runtimes. The backend streams tool activity and generated responses to the frontend using Server-Sent Events (SSE).

---

## 🔌 MCP Tools

Qonnect exposes six tools that allow the agent to interact with the optimization environment.

| Tool                 | Capability                                 |
| -------------------- | ------------------------------------------ |
| `formulate_problem`  | Formulates an optimization problem         |
| `solve_classical`    | Executes a classical optimization approach |
| `run_qaoa`           | Runs an experimental QAOA workflow         |
| `hybrid_shortlist`   | Supports hybrid optimization workflows     |
| `analyze_results`    | Analyzes optimization results              |
| `estimate_resources` | Estimates required computational resources |

**Why MCP?**

Instead of embedding optimization logic directly into the agent, Qonnect uses MCP as a standardized interface between reasoning and execution.

This modular approach makes it possible to extend the system with additional tools and optimization strategies.

---

## 🧰 Technology Stack

<p align="center">
&#x20; <img src="[https://skillicons.dev/icons?i=python,react,typescript,vite,aws](https://skillicons.dev/icons?i=python,react,typescript,vite,aws)" alt="Technology stack">
</p>

| Layer                   | Technologies                       |
| ----------------------- | ---------------------------------- |
| Agentic AI              | Strands Agents, Amazon Nova 2 Lite |
| Tool integration        | Model Context Protocol             |
| Cloud infrastructure    | AWS Lambda, AWS Amplify            |
| Agent infrastructure    | Amazon Bedrock AgentCore           |
| Quantum experimentation | Amazon Braket                      |
| Backend                 | Python 3.13, FastAPI               |
| Frontend                | React, TypeScript, Vite            |
| Streaming               | Server-Sent Events                 |

---

## 📂 Project Structure

```text
qonnect/
├── app/
│   ├── chat_backend/   # API and Lambda integration
│   ├── quantum_agent/  # Strands AI agent
│   └── quantum_mcp/    # MCP server and optimization tools
│
├── web/                # React + TypeScript frontend
│
├── assets/
│   ├── qonnect-demo.gif
│   └── qonnect-architecture.svg
│
└── README.md
```

---

## 🚧 Project Status

<kbd>Working end-to-end experimental demo</kbd>

- [x] Conversational frontend deployed on AWS Amplify
- [x] Lambda Function URL integration
- [x] FastAPI backend on AWS Lambda
- [x] Strands agent on AgentCore Runtime
- [x] Independent MCP server on AgentCore Runtime
- [x] Six MCP optimization tools
- [x] Classical optimization workflow
- [x] Experimental QAOA integration with Amazon Braket
- [x] AgentCore Memory integration
- [x] SSE streaming between backend and frontend
- [ ] Further testing, evaluation, and optimization

> **Research note:** Qonnect is an experimental project. It does not claim quantum advantage or speedup. Its purpose is to explore how classical and quantum optimization approaches can be orchestrated and evaluated through AI agents.

---

## 🔭 Vision

Qonnect explores the intersection of three areas:

- **Agentic AI:** agents that can reason, select tools, and coordinate workflows.
- **Cloud computing:** infrastructure that makes emerging technologies accessible.
- **Quantum computing:** experimentation with optimization approaches and quantum resources.

Rather than replacing classical computing, the project explores how AI agents can help users work with different computational approaches and understand when each might be useful.

---

## 👩‍💻 Author

**Joselyn Lagunas**

AWS Community Builder · Agentic AI · Cloud Computing · Emerging Technologies

[LinkedIn](https://www.linkedin.com/in/joselyn-lagunas/)

---

<p align="center">
&#x20; <strong>Qonnect</strong><br>
&#x20; <em>Connecting ideas, agents, and emerging computational technologies.</em>
</p>

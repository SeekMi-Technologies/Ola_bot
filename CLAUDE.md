# CLAUDE.md — NanoBot (Ola AI Backend)

This file provides guidance to Claude Code when working with code in this repository.

## Project

NanoBot is Ola Technologies' AI agent framework — a **Python** microservice (v0.1.5.post2) that powers the "Ask Ola" chat feature in the Ola CRM. It is a fork of [HKUDS/nanobot](https://github.com/HKUDS/nanobot) v0.1.4.post6 with Ola-specific customizations.

**Language & Runtime:** Python 3.11+, managed with `pyproject.toml` (hatchling build).

**Deployment model:** Runs as a separate Docker service alongside Ola CRM. CRM's `olaController` HTTP-proxies `/api/ola/chat` to NanoBot's serve endpoint.

## Common Commands

```bash
# Install (dev mode)
pip install -e ".[dev]"

# Run CLI
nanobot serve                    # Start API server
nanobot chat                     # Interactive CLI chat

# Tests
pytest                           # Run all tests
pytest tests/test_xxx.py         # Single file
pytest --cov=nanobot             # With coverage

# Lint
ruff check nanobot/              # Lint
ruff format nanobot/             # Format
```

## Architecture

### Core (`nanobot/`)

```
nanobot/
├── __init__.py          # Version
├── __main__.py          # Entry point
├── nanobot.py           # Facade / main class
├── agent/               # Core agent loop
│   ├── loop.py          # Agent event loop
│   ├── context.py       # Context management
│   ├── memory.py        # Two-layer memory (SOUL.md, USER.md, MEMORY.md, history.jsonl)
│   ├── runner.py        # Agent runner
│   ├── subagent.py      # Sub-agent orchestration
│   └── tools/           # Built-in tools (filesystem, search, mcp, etc.)
├── api/                 # HTTP/WebSocket API server
├── bus/                 # Event bus
├── channels/            # 13+ chat channels (WhatsApp, WeChat, Email, Telegram, Slack, etc.)
├── cli/                 # CLI commands (typer-based)
├── command/             # Command processing
├── config/              # Configuration management
├── cron/                # Scheduled tasks
├── heartbeat/           # Health monitoring
├── providers/           # LLM providers (OpenAI, Anthropic native SDKs)
├── security/            # Security utilities
├── session/             # Session management
├── skills/              # Built-in skills (memory, cron, github, summarize, tmux, weather, etc.)
├── templates/           # Default workspace templates (SOUL.md, USER.md, etc.)
├── utils/               # Utility functions
└── web/                 # Web UI assets
```

### Bridge (`bridge/`)
Node.js/TypeScript bridge for WhatsApp (Baileys). Compiled separately.

### Workspace (`~/.nanobot/` or custom path)
Runtime workspace containing per-agent configuration, skills, memory files, and history.

## Key Design Principles

- **Do NOT rewrite in Node.js** — the decision is to keep NanoBot as a Python microservice.
- **No silent errors** — every `except` must log or re-raise; never `except: pass`.
- **Memory system** uses `SOUL.md` (persona), `USER.md` (user profile), `MEMORY.md` (facts), and `history.jsonl`.
- **Skills** are defined as `SKILL.md` files with YAML frontmatter + Markdown instructions.
- **MCP** is the planned standard interface — CRM backend exposes business actions as MCP tools that NanoBot can call.
- **Providers** use native OpenAI/Anthropic SDKs (litellm was removed in v0.1.4).

## Integration with Ola CRM

- CRM's `olaController` proxies chat requests to NanoBot's serve endpoint
- MCP tools allow NanoBot to perform CRM actions (quote creation, product lookup, etc.)
- Configuration template at `../Ola/ola/nanobot.config.template.json`
- Workspace files at `../Ola/ola/nanobot-workspace/` (SOUL.md, AGENTS.md, TOOLS.md, USER.md)

## Hard Boundaries

- ❌ No silent errors in any exception handler
- ❌ No hardcoded secrets or API keys
- ❌ Never drop or truncate conversation history silently
- ❌ Do not modify `bridge/` without understanding the WhatsApp Baileys integration
- ✅ Follow existing code style (ruff: line-length=100, target py311)
- ✅ Use `loguru` for all logging
- ✅ Use `pydantic` for data validation
- ✅ All new tools must have proper error handling and type hints

## Git Workflow

- Branch: `ola-dev` (main development branch)
- Remote: `origin` → SeekMi-Technologies fork
- Follow the same SDD discipline as Ola CRM when working across both repos
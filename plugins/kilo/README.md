# Ensemble — Kilo Code (and Cline / Roo / Continue / Cursor)

Ensemble ships as a standard MCP server, so it works in Kilo Code and any other
MCP-capable client. There's nothing Kilo-specific in the engine — just config.

## Install

```bash
pip install "ensemble[mcp]"     # provides the `ensemble-mcp` command
```

Set at least two of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`.

## Add the server to Kilo Code

**Option A — config file.** Copy [`kilo.jsonc`](./kilo.jsonc) to
`~/.config/kilo/kilo.jsonc` (global) or `.kilo/kilo.jsonc` (this project), and
fill in your keys.

**Option B — UI.** Settings → MCP → Add Server → Local (stdio):
- Command: `ensemble-mcp`
- Env: your API keys
- **Timeout: raise it to ~600000ms** — the default 10s will abort a debate.

Once connected, the `ensemble_debate` tool is available to the agent. Ask Kilo
to "convene the council on …" or wire it into a custom **Mode** for one-click
access.

## Other clients

The same `ensemble-mcp` command works in **Cline, Roo Code, Continue, Cursor,
and VS Code Copilot** — add it via each client's MCP settings. (For the Claude
Code packaging with a slash command, see [`../claude-code/`](../claude-code/).)

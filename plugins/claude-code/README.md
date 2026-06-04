# Ensemble — Claude Code plugin

Adds a multi-model **council** to Claude Code: the `ensemble_debate` MCP tool
(LLMs propose → peer-review → rebut → vote → synthesize → converge) and an
`/ensemble` slash command.

## Prerequisites

Install the engine + MCP server, and set at least two provider keys:

```bash
pip install "ensemble[mcp]"          # provides the `ensemble-mcp` command
export OPENAI_API_KEY=...            # any two of the three
export ANTHROPIC_API_KEY=...
export DEEPSEEK_API_KEY=...
```

> No global install? Swap the command in `.mcp.json` for
> `"command": "uvx", "args": ["--from", "ensemble[mcp]", "ensemble-mcp"]`.

## Install the plugin

From the marketplace at the repo root:

```text
/plugin marketplace add raiyanyahya/ensemble      # or a local path to this repo
/plugin install ensemble@ensemble
```

Then restart Claude Code. You should see the `ensemble` MCP server connect and
an `/ensemble` command appear.

## Use

- **Slash command:** `/ensemble Should we use Postgres or DynamoDB for this workload?`
- **Tool (Claude calls it directly):** ask Claude to "get the council's opinion on …";
  it will invoke `ensemble_debate` (use `quick: false` for a deep debate).

Debates are written to `~/.ensemble/debates/<id>/` for inspection.

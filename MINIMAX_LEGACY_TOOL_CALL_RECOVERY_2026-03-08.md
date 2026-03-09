# MiniMax Legacy Tool Call Recovery

## What Changed

- Added recovery for legacy MiniMax bracketed tool-call markup in [GPTPlugins4All/assistant.py](/Users/trevormartin/Projects/laserreach/GPTPlugins4All/GPTPlugins4All/assistant.py).
- Hardened both sync and streaming old-mode loops so recovered legacy tool calls are executed instead of leaking as plain assistant text.
- Hardened `execute_function(...)` to return structured errors instead of crashing when a handler or config is missing.
- Increased the direct MiniMax default tool-round budget from `12` to `18` so longer research + outreach workflows can finish without an immediate budget stop.

## Root Cause

Direct MiniMax sometimes emits literal `[TOOL_CALL]...[/TOOL_CALL]` blocks instead of native tool-call objects. The old-mode path treated those blocks as regular text, which let the model “say” the tool call without actually executing it. On long manual-targeting turns, the direct MiniMax tool-round limit was also too low, so the agent could reach dossier prep and then stop before outreach/slack completion.

## Why This Is Safe

- The recovery path only activates when native tool calls are absent and the response contains the legacy bracketed format.
- Native tool-call handling is unchanged.
- The higher MiniMax tool-round default only applies to direct MiniMax unless the environment explicitly overrides it.

## Validation

- `pytest -q /Users/trevormartin/Projects/laserreach/GPTPlugins4All/tests/test_minimax_legacy_tool_calls.py`

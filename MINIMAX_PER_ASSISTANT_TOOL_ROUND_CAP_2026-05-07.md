# MiniMax Per-Assistant Tool Round Cap - 2026-05-07

## Root Cause

GPTPlugins previously allowed tool-round limits only through process-global environment variables. ABM needs a safer MiniMax loop budget per assistant run so a bad or confused workflow cannot keep calling tools up to the high global MiniMax default.

## Change

Added `Assistant(max_tool_rounds=...)`. When provided, `_max_tool_rounds()` uses the instance value before checking environment defaults.

## Verification

- `pytest -q tests/test_minimax_legacy_tool_calls.py`: passed.

## Remaining Risk

This caps sequential tool rounds inside one assistant response. It does not cap how many separate agent runs the application starts.

# MiniMax Tool Budget Hardening

Date: 2026-03-09
Branch: `codex/manual-targeting-skill`

## Root Cause

Direct MiniMax execution was still using a conservative default tool-round budget for long multi-step ABM workflows. That made it easier for the model to stop short on real manual-targeting runs even though the token cost remained negligible.

## What Changed

Updated [assistant.py](/Users/trevormartin/Projects/laserreach/GPTPlugins4All/GPTPlugins4All/assistant.py) so the default direct-MiniMax tool-round budget increases from `18` to `40`.

Updated [test_minimax_legacy_tool_calls.py](/Users/trevormartin/Projects/laserreach/GPTPlugins4All/tests/test_minimax_legacy_tool_calls.py) to lock that behavior in.

## Validation

- `pytest -q tests/test_minimax_legacy_tool_calls.py`

Passed.

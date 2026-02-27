# Search Tool Fix (Serper + Fallback) - 2026-02-27

## What Was Broken

`search_google` in `GPTPlugins4All/assistant.py` depended on a brittle `googlesearch` scraping path that now frequently returns empty results or errors.

## Root Cause

1. Single-provider dependency on `googlesearch` with no resilient fallback.
2. Google HTML/search anti-bot changes made the old approach unreliable.
3. Packaging pinned a Git-based `googlesearch` fork, which is fragile for installs.

## What Changed

1. Replaced `search_google` internals with provider failover:
   - Primary: Serper API (`SERPER_API_KEY`, optional `SERPER_API_URL`)
   - Fallback: DuckDuckGo HTML search (free, no API key)
   - Last fallback: legacy `googlesearch` if present
2. Kept tool name `search_google` for backward compatibility.
3. Updated built-in tool descriptions to reflect generic web search behavior.
4. Made legacy Google import optional (`try/except`) so missing dependency does not break module import.
5. Removed Git-pinned `googlesearch` dependency from `setup.py`.
6. Added unit tests for provider selection, fallback behavior, and empty-query guard.
7. Updated README with provider configuration.

## Why This Is Safe

1. External interface is unchanged (`search_google` still exists and same input schema is used).
2. Failover behavior only broadens availability; it does not alter unrelated assistant/tool logic.
3. Unit tests cover success path, fallback path, and error guardrails.

## Validation Matrix

1. Unit
   - `pytest -q GPTPlugins4All/tests/test_search_provider.py`
   - Result: `4 passed`

2. Integration
   - Live call: `assistant.search_google("openai", num_results=3)` with no Serper key
   - Result: Serper attempt failed as expected (no key), auto-fallback to DuckDuckGo returned results

3. Regression
   - Full backend test suite in consumer app: `cd laserreach && pytest -q`
   - Result: completed successfully (exit code 0, with existing skipped tests)

4. Live dev smoke
   - Tool dispatch path: `Assistant.execute_function("search_google", "{\"query\":\"site reliability engineering\"}")`
   - Result: returned non-error search output via fallback provider

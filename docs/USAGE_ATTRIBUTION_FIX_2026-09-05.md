# Usage attribution fix (2026-09-05)

## What changed

Usage events now attribute tokens and estimator callbacks to the model on the completion or request payload. Streaming terminal chunks use the chunk model when supplied and otherwise inherit the request model.

## Root cause

The event emitter previously always used `Assistant.model`, which can differ from a per-request model override or provider fallback.

## Verification

- The request and streaming regression tests fail on the previous source (two failed, one passed) and pass with this fix.
- Full shared-library suite: 16 passed, zero skipped, using a dummy OpenAI key and mocked provider calls.
- Dispatch tests recovered from the original checkout cover existing remote-master behavior; this change does not modify dispatch authorization.
- A test-only CI workflow runs the suite on pushes and pull requests. It installs the existing runtime imports, including python-dotenv, and cannot publish a package.

## Remaining gap

Provider pricing values remain the existing configured heuristics and were intentionally unchanged.

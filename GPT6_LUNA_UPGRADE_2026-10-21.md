# GPT-6 Luna assistant support (2026-10-21)

The shared assistant handled GPT-5 model names when choosing token limits and
function-tool reasoning. A direct switch to GPT-6 Luna would have sent the
wrong token parameter and could reject Chat Completions tool calls.

The assistant now recognizes GPT-6 in those paths, uses GPT-6 Luna for image
fallback, and estimates its input cost at $0.10 per million tokens. The GPT-5.6
paths remain available for saved configurations. Both Luna versions explicitly
use `reasoning_effort=none` for Chat Completions function tools.

Validation: 14 local tests passed, including parameter and cost regression
checks. A live synthetic function-tool roundtrip returned `SMOKE_OK`. The
Python 3.11 pull-request test job passed for commit `2443b52`.

This nested repository sets its Codex child-agent default to GPT-6 Luna and
overrides the older user-level OMX model guidance when opened on its own.

Price source: [OpenAI GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna).

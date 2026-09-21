# Luna Function Tool Reasoning Compatibility

## Root cause

GPT-5.6 Luna rejects Chat Completions requests that include function tools unless
`reasoning_effort` is explicitly set to `"none"`. The shared payload builder left
the field absent when callers omitted it and allowed non-`"none"` overrides,
including the `openai/gpt-5.6-luna` model alias.

## Fix

`Assistant._prepare_chat_payload` now forces `reasoning_effort="none"` only for
the exact `gpt-5.6-luna` and `openai/gpt-5.6-luna` request models when the
prepared tool list contains a function tool. Requests without function tools
and requests for other models retain their existing payload values.

## Testing

Regression coverage checks omitted/default values, non-`"none"` overrides, the
OpenAI model alias, no-tool requests, non-function tools, other models, and a
streaming tool follow-up payload.

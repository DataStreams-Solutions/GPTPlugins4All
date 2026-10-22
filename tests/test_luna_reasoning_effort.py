from GPTPlugins4All.assistant import Assistant
import pytest


FUNCTION_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_contact",
        "description": "Look up a contact.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _assistant(model="gpt-6-luna"):
    assistant = object.__new__(Assistant)
    assistant.model = model
    assistant.chat_base_url = ""
    return assistant


def _payload(model, *, tools=None, reasoning_effort="__omitted__"):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Find the contact."}],
        "tools": tools,
    }
    if reasoning_effort != "__omitted__":
        payload["reasoning_effort"] = reasoning_effort
    return payload


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-6-luna"])
def test_luna_function_tools_default_reasoning_effort_is_explicitly_none(model):
    payload = _assistant()._prepare_chat_payload(
        _payload(model, tools=[FUNCTION_TOOL])
    )

    assert payload["reasoning_effort"] == "none"


@pytest.mark.parametrize("model", ["openai/gpt-5.6-luna", "openai/gpt-6-luna"])
def test_luna_alias_function_tools_override_reasoning_effort_to_none(model):
    payload = _assistant()._prepare_chat_payload(
        _payload(model, tools=[FUNCTION_TOOL], reasoning_effort="high")
    )

    assert payload["reasoning_effort"] == "none"


def test_luna_function_tools_preserve_non_function_tools_without_reasoning_guard():
    payload = _assistant()._prepare_chat_payload(
        _payload("gpt-5.6-luna", tools=[{"type": "file_search"}], reasoning_effort="high")
    )

    assert payload["reasoning_effort"] == "high"


def test_luna_without_tools_preserves_reasoning_effort():
    payload = _assistant()._prepare_chat_payload(
        _payload("gpt-5.6-luna", tools=None, reasoning_effort="high")
    )

    assert "tools" not in payload
    assert payload["reasoning_effort"] == "high"


def test_other_models_with_function_tools_preserve_reasoning_effort():
    payload = _assistant()._prepare_chat_payload(
        _payload("gpt-5.6-sol", tools=[FUNCTION_TOOL], reasoning_effort="high")
    )

    assert payload["reasoning_effort"] == "high"


def test_luna_streaming_followup_with_function_tools_forces_none():
    payload = _payload("gpt-5.6-luna", tools=[FUNCTION_TOOL], reasoning_effort="medium")
    payload.update(
        stream=True,
        messages=[
            {"role": "user", "content": "Find the contact."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup_contact", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ],
    )

    prepared = _assistant()._prepare_chat_payload(payload)

    assert prepared["stream"] is True
    assert prepared["reasoning_effort"] == "none"


def test_non_streaming_drops_stream_options_from_shared_defaults():
    payload = _payload("gpt-5.6-luna", tools=[FUNCTION_TOOL])
    payload["stream_options"] = {"include_usage": True}
    prepared = _assistant()._prepare_chat_payload(payload)
    assert "stream_options" not in prepared
    assert payload["stream_options"] == {"include_usage": True}


def test_streaming_preserves_usage_options():
    payload = _payload("gpt-5.6-luna", tools=[FUNCTION_TOOL])
    payload.update(stream=True, stream_options={"include_usage": True})
    prepared = _assistant()._prepare_chat_payload(payload)
    assert prepared["stream_options"] == {"include_usage": True}


def test_gpt6_luna_uses_completion_token_parameter_and_current_cost():
    assistant = _assistant()
    assistant.chat_completion_defaults = {}
    assistant.max_completion_tokens = 256
    assistant.max_tokens = None

    payload = assistant._build_chat_data({"model": "gpt-6-luna"})

    assert payload["max_completion_tokens"] == 256
    assert "max_tokens" not in payload
    assert assistant._image_fallback_model() == "gpt-6-luna"
    assert assistant._estimate_input_cost_usd("gpt-6-luna", 1_000_000) == 0.10

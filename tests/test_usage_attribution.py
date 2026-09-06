from types import SimpleNamespace

from GPTPlugins4All.assistant import Assistant


def _assistant(events, *, streaming=False):
    assistant = Assistant(
        configs=[],
        name="test",
        instructions="test",
        model="gpt-5.6-sol",
        openai_key="test-key",
        old_mode=True,
        streaming=streaming,
        get_thread=lambda _tid: {"messages": []},
        put_thread=lambda _tid, _messages: None,
        event_listener=lambda event: events.append(event),
        provider_name="openai",
        usage_cost_estimator=lambda model, usage: {"selected_model": model},
    )
    return assistant


def _usage():
    return SimpleNamespace(prompt_tokens=10, completion_tokens=2)


def test_nonstream_callsite_uses_payload_model_override():
    events = []
    assistant = _assistant(events)
    completion = SimpleNamespace(usage=_usage(), choices=[])

    class Completions:
        def create(self, **kwargs):
            assert kwargs["model"] == "gpt-5.6-luna"
            return completion

    assistant.openai_client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions())
    )
    assistant._ensure_image_model_compatibility = lambda payload: assistant.openai_client
    assistant._prepare_chat_payload = lambda payload: payload

    assistant._chat_completion_create_with_context_recovery(
        {"model": "gpt-5.6-luna", "messages": [{"role": "user", "content": "hi"}]}
    )

    usage_event = next(event for event in events if event["type"] == "chat_completion_usage")
    assert usage_event["model"] == "gpt-5.6-luna"
    assert usage_event["selected_model"] == "gpt-5.6-luna"


def test_stream_callsite_uses_chunk_model_override():
    events = []
    assistant = _assistant(events, streaming=True)
    captured_payloads = []
    chunks = iter([
        SimpleNamespace(model="openai/gpt-5.6-luna", usage=_usage(), choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(content="done", tool_calls=None, reasoning=None),
            )]
        ),
    ])

    def fake_completion(payload):
        captured_payloads.append(payload)
        return chunks

    assistant._chat_completion_create_with_context_recovery = fake_completion
    result = list(assistant.handle_old_mode_streaming("hi"))

    assert result == ["done"]
    assert captured_payloads[0]["model"] == "gpt-5.6-sol"
    assert events[0]["model"] == "openai/gpt-5.6-luna"
    assert events[0]["selected_model"] == "openai/gpt-5.6-luna"


def test_stream_callsite_falls_back_to_request_model_when_chunk_omits_model():
    events = []
    assistant = _assistant(events, streaming=True)
    assistant._chat_completion_create_with_context_recovery = lambda _payload: iter([
        SimpleNamespace(usage=_usage(), choices=[]),
        SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(content="done", tool_calls=None, reasoning=None),
            )]
        ),
    ])

    assert list(assistant.handle_old_mode_streaming("hi")) == ["done"]
    assert events[0]["model"] == "gpt-5.6-sol"

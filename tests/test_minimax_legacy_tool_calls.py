from __future__ import annotations

from types import SimpleNamespace

from GPTPlugins4All.assistant import Assistant


def _make_assistant(monkeypatch, *, streaming: bool = False) -> tuple[Assistant, list[dict], list[dict]]:
    thread_store: dict[str, dict] = {"thread-test": {"messages": []}}
    executed: list[dict] = []
    events: list[dict] = []

    monkeypatch.setattr(
        Assistant,
        "create_assistant_and_thread",
        lambda self, **_kwargs: (SimpleNamespace(id="asst_test"), SimpleNamespace(id="thread-test")),
    )

    assistant = Assistant(
        configs=[],
        name="test",
        instructions="Test assistant",
        model="MiniMax-M2.5",
        thread_id="thread-test",
        openai_key="test-key",
        base_url="https://api.minimaxi.chat/v1",
        old_mode=True,
        streaming=streaming,
        other_tools=[
            {
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "description": "Dummy tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"company_id": {"type": "string"}},
                        "required": ["company_id"],
                    },
                },
            }
        ],
        other_functions={},
        get_thread=lambda thread_id: {
            "messages": list((thread_store.get(thread_id) or {}).get("messages", []))
        },
        put_thread=lambda thread_id, messages: thread_store.__setitem__(
            thread_id,
            {"messages": list(messages)},
        ),
        event_listener=lambda row: events.append(dict(row or {})),
        emit_tool_preamble=False,
    )

    assistant.execute_function = lambda tool_name, tool_args, _user_tokens=None: executed.append(
        {"tool_name": tool_name, "tool_args": tool_args}
    ) or {"success": True, "tool_name": tool_name}
    return assistant, executed, events


def test_sync_chat_recovers_legacy_tool_call_markup(monkeypatch):
    assistant, executed, events = _make_assistant(monkeypatch, streaming=False)
    completions = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=(
                            '[TOOL_CALL]\n'
                            '{tool => "dummy_tool", args => {\n'
                            '  --company_id "comp_123"\n'
                            '}}\n'
                            '[/TOOL_CALL]'
                        ),
                        tool_calls=None,
                        reasoning=None,
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content='{"status":"done"}',
                        tool_calls=None,
                        reasoning=None,
                    )
                )
            ]
        ),
    ]

    monkeypatch.setattr(
        assistant,
        "_chat_completion_create_with_context_recovery",
        lambda _payload: completions.pop(0),
    )

    response = assistant.get_assistant_response("run the tool")

    assert response == '{"status":"done"}'
    assert executed == [{"tool_name": "dummy_tool", "tool_args": '{"company_id": "comp_123"}'}]
    assert events
    assert events[0]["tool_name"] == "dummy_tool"


def test_stream_chat_recovers_legacy_tool_call_markup(monkeypatch):
    assistant, executed, events = _make_assistant(monkeypatch, streaming=True)

    first_stream = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(
                            content='[TOOL_CALL]\n{tool => "dummy_tool", args => {\n  --company_id "comp_999"\n}}\n[/TOOL_CALL]',
                            tool_calls=None,
                            reasoning=None,
                        ),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="", tool_calls=None, reasoning=None))]
            ),
        ]
    )
    second_stream = iter(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=None,
                        delta=SimpleNamespace(content='{"status":"stream_done"}', tool_calls=None, reasoning=None),
                    )
                ]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(finish_reason="stop", delta=SimpleNamespace(content="", tool_calls=None, reasoning=None))]
            ),
        ]
    )
    completions = [first_stream, second_stream]

    monkeypatch.setattr(
        assistant,
        "_chat_completion_create_with_context_recovery",
        lambda _payload: completions.pop(0),
    )

    response = "".join(list(assistant.get_assistant_response("run the tool")))

    assert response == '{"status":"stream_done"}'
    assert executed == [{"tool_name": "dummy_tool", "tool_args": '{"company_id": "comp_999"}'}]
    assert events
    assert events[0]["tool_name"] == "dummy_tool"


def test_minimax_defaults_to_higher_tool_round_budget(monkeypatch):
    assistant, _executed, _events = _make_assistant(monkeypatch, streaming=False)
    monkeypatch.delenv("ASSISTANT_MAX_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("ASSISTANT_MAX_TOOL_ROUNDS_MINIMAX", raising=False)
    monkeypatch.delenv("ASSISTANT_MAX_TOOL_ROUNDS_DEFAULT", raising=False)
    assert assistant._max_tool_rounds() == 40

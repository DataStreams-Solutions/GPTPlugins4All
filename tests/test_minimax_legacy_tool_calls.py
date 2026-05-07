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
        embedding_key="embedding-key",
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
    assert assistant._max_tool_rounds() == 80

def test_tool_output_context_strips_transport_and_run_memory_noise():
    assistant = object.__new__(Assistant)
    assistant.tool_output_context_max_chars = 1200
    noisy = {
        "success": True,
        "person_id": "linkedin_123",
        "person": {
            "tenant_id": "orgl",
            "person_id": "linkedin_123",
            "company_id": "18914703",
            "company_name": "DataStreams Solutions",
            "created_at": "2026-05-05T13:10:14.566632",
            "email": "trevor@datastreamssolutions.com",
            "headline": "Founder/CEO DataStreams Solutions",
            "last_manual_targeting_seen_at": "2026-05-05T13:10:14.536057",
            "linkedin_url": "https://www.linkedin.com/in/trevor-martin-86567859",
            "name": "Trevor Martin",
            "phone_number": "+1 262-404-7897",
            "provider_id": "5ae95a3ea6da98eae4e2bef7",
            "source": "manual_targeting_discovery",
            "title": "Founder/CEO",
            "tracking_stage": "discovered",
            "updated_at": "2026-05-05T13:13:14.089211",
        },
        "_run_memory": {
            "run_id": "run_chat_123",
            "paths": {"ledger": "agent_runs/run_chat_123/ledger.jsonl"},
            "next_action": "Resolve tool failure: account_not_allowed",
        },
    }

    compacted = assistant._compact_tool_outputs_for_context([
        {
            "tool_call_id": "call_abc",
            "tool_name": "update_person",
            "tool_arguments": '{"person_id":"linkedin_123"}',
            "output": assistant._json_dumps_safe(noisy),
        }
    ])
    visible = assistant._model_visible_tool_outputs(compacted)
    rendered = assistant._tool_output_followup_hint(compacted)

    assert visible == [
        {
            "tool": "update_person",
            "result": {
                "success": True,
                "person_id": "linkedin_123",
                "person": {
                    "person_id": "linkedin_123",
                    "name": "Trevor Martin",
                    "title": "Founder/CEO",
                    "headline": "Founder/CEO DataStreams Solutions",
                    "company_id": "18914703",
                    "company_name": "DataStreams Solutions",
                    "email": "trevor@datastreamssolutions.com",
                    "phone_number": "+1 262-404-7897",
                    "linkedin_url": "https://www.linkedin.com/in/trevor-martin-86567859",
                    "tracking_stage": "discovered",
                },
            },
        }
    ]
    assert "tool_call_id" not in rendered
    assert "tool_arguments" not in rendered
    assert "_run_memory" not in rendered
    assert "last_manual_targeting_seen_at" not in rendered
    assert "tenant_id" not in rendered
    assert "ledger.jsonl" not in rendered


def test_payload_estimate_emits_tool_schema_tokens():
    events = []
    assistant = object.__new__(Assistant)
    assistant.model = "MiniMax-M2.7"
    assistant.event_listener = lambda row: events.append(dict(row or {}))
    assistant._chat_payload_estimate_seq = 0

    payload = {
        "model": "MiniMax-M2.7",
        "messages": [{"role": "system", "content": "hello"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "dummy_tool",
                    "description": "A dummy tool",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
            }
        ],
    }

    assistant._emit_chat_payload_estimate(payload, phase="unit")

    assert events
    event = events[0]
    assert event["type"] == "chat_payload_estimate"
    assert event["input_tokens"] > event["message_tokens"]
    assert event["tool_schema_tokens"] > 0
    assert event["estimated_input_cost_usd"] > 0


def test_max_tool_rounds_can_be_set_per_assistant(monkeypatch):
    assistant, _executed, _events = _make_assistant(monkeypatch, streaming=False)
    assert assistant._max_tool_rounds() == 80

    capped = Assistant(
        configs=[],
        name="capped",
        instructions="Test assistant",
        model="MiniMax-M2.7",
        thread_id="thread-test",
        openai_key="test-key",
        embedding_key="embedding-key",
        base_url="https://api.minimaxi.chat/v1",
        old_mode=True,
        max_tool_rounds=24,
        get_thread=lambda _thread_id: {"messages": []},
        put_thread=lambda _thread_id, _messages: None,
    )
    assert capped._max_tool_rounds() == 24

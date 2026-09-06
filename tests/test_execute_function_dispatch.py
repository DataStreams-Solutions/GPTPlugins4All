from GPTPlugins4All.assistant import Assistant


def _build_assistant(*, other_tools=None, other_functions=None):
    assistant = object.__new__(Assistant)
    assistant.configs = []
    assistant.other_tools = list(other_tools or [])
    assistant.other_functions = dict(other_functions or {})
    assistant.multiple_configs = False
    assistant.search_window = 1000
    return assistant


def test_execute_function_dispatches_local_handler_without_tool_registration():
    seen = {}

    def handler(payload):
        seen.update(payload)
        return {"success": True, "path": payload.get("path")}

    assistant = _build_assistant(
        other_tools=[],
        other_functions={"operator_get_workspace_file": handler},
    )

    result = assistant.execute_function(
        "operator_get_workspace_file",
        '{"account_id": "acct_1", "path": "notes/a.md"}',
    )

    assert result["success"] is True
    assert seen == {"account_id": "acct_1", "path": "notes/a.md"}


def test_execute_function_resolves_operator_alias_for_workspace_file_tool():
    calls = []

    def handler(payload):
        calls.append(dict(payload or {}))
        return {"success": True, "alias_used": True}

    assistant = _build_assistant(
        other_tools=[],
        other_functions={"operator_get_workspace_file": handler},
    )

    result = assistant.execute_function(
        "get_workspace_file",
        '{"account_id": "acct_1", "path": "dossiers/company/acme.md"}',
    )

    assert result == {"success": True, "alias_used": True}
    assert calls == [{"account_id": "acct_1", "path": "dossiers/company/acme.md"}]


def test_execute_function_resolves_route_style_alias_to_operator_handler():
    calls = []

    def handler(payload):
        calls.append(dict(payload or {}))
        return {"success": True, "route_alias_used": True}

    assistant = _build_assistant(
        other_tools=[],
        other_functions={"operator_get_workspace_file": handler},
    )

    result = assistant.execute_function(
        "get_workspace_file_route_get",
        '{"account_id": "acct_1", "path": "dossiers/company/acme.md"}',
    )

    assert result == {"success": True, "route_alias_used": True}
    assert calls == [{"account_id": "acct_1", "path": "dossiers/company/acme.md"}]


def test_execute_function_returns_function_not_found_when_no_dispatch_path_exists():
    assistant = _build_assistant(other_tools=[], other_functions={})

    result = assistant.execute_function("missing_tool", '{"x": 1}')

    assert result["success"] is False
    assert result["error"] == "function_not_found"
    assert result["function_name"] == "missing_tool"
    assert "operator_missing_tool" in result["tried_local_names"]
    assert result["configured_api_count"] == 0

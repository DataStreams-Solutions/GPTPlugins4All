# `get_workspace_file` dispatch regression (2026-03-23)

The existing `execute_function(...)` path resolves local handlers from `other_functions`, including operator and route aliases, and reports `function_not_found` when no callable or API configuration exists. The regression tests cover direct, operator-alias, route-alias, and missing-handler cases.

Local handler registration remains the caller's responsibility: `other_functions` must contain only handlers that the assistant is authorized to execute. This test documents dispatch behavior; it does not expand the authorization policy or treat an unregistered callable as safe by itself.

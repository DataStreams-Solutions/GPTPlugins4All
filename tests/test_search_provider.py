import sys
from pathlib import Path

import requests

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "GPTPlugins4All"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import assistant  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


def test_search_google_prefers_serper_when_available(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("GPTPLUGINS_SEARCH_PROVIDER", raising=False)

    def fake_post(url, json, headers, timeout):
        assert "serper" in url
        assert json["q"] == "openai"
        return FakeResponse(
            status_code=200,
            json_data={
                "organic": [
                    {"link": "https://example.com/a", "title": "A", "snippet": "alpha"},
                    {"link": "https://example.com/b", "title": "B", "snippet": "beta"},
                ]
            },
        )

    def fail_get(*args, **kwargs):
        raise AssertionError("DuckDuckGo fallback should not run when Serper succeeds")

    monkeypatch.setattr(assistant.requests, "post", fake_post)
    monkeypatch.setattr(assistant.requests, "get", fail_get)

    result = assistant.search_google("openai", num_results=2)
    assert "Results from web search (serper): openai" in result
    assert "https://example.com/a - A - alpha" in result


def test_search_google_falls_back_to_duckduckgo_without_serper_key(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    monkeypatch.delenv("GPTPLUGINS_SEARCH_PROVIDER", raising=False)

    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2F">Example</a>
        <a class="result__snippet">Snippet text</a>
      </div>
    </body></html>
    """

    def fake_get(*args, **kwargs):
        return FakeResponse(status_code=200, text=html)

    monkeypatch.setattr(assistant.requests, "get", fake_get)

    result = assistant.search_google("fallback query")
    assert "Results from web search (duckduckgo): fallback query" in result
    assert "https://example.org/ - Example - Snippet text" in result


def test_search_google_falls_back_when_serper_errors(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.delenv("GPTPLUGINS_SEARCH_PROVIDER", raising=False)

    def failing_post(*args, **kwargs):
        return FakeResponse(status_code=500, text="server error")

    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="https://example.net">Example Net</a>
        <div class="result__snippet">Net snippet</div>
      </div>
    </body></html>
    """

    def fake_get(*args, **kwargs):
        return FakeResponse(status_code=200, text=html)

    monkeypatch.setattr(assistant.requests, "post", failing_post)
    monkeypatch.setattr(assistant.requests, "get", fake_get)

    result = assistant.search_google("resilient search")
    assert "Results from web search (duckduckgo): resilient search" in result
    assert "https://example.net - Example Net - Net snippet" in result


def test_search_google_rejects_empty_query():
    result = assistant.search_google("   ")
    assert result == "Error: query is required"

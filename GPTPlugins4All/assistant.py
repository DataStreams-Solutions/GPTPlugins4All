import os
import time
import json
import ast
from dotenv import load_dotenv
import uuid
import threading
import requests
from bs4 import BeautifulSoup
try:
    from googlesearch import search as google_search
except Exception:
    google_search = None
import base64
import copy
from datetime import datetime
import logging
import re
from urllib.parse import parse_qs, unquote, urlparse

load_dotenv()

from typing import Optional
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

logger = logging.getLogger(__name__)

cost_dict = {'o3-mini': 110/1000000, 'gpt-3.5-turbo': 1/1000000, 'o1': 60000/1000000, 'gpt-4o': 1000/1000000, 'gpt-4o-mini': 60/1000000}
SEARCH_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
OPENAI_CHAT_TOOL_LIMIT = 128


def _format_search_results(query, provider, results):
    lines = []
    for item in results:
        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        if not any([url, title, description]):
            continue
        lines.append(f"{url} - {title} - {description}")
    if not lines:
        return "No search results found for: " + query
    return f"Results from web search ({provider}): {query}\n" + "\n\n".join(lines)


def _normalize_duckduckgo_redirect_url(url):
    if not url:
        return ""
    normalized = str(url).strip()
    if normalized.startswith("//"):
        normalized = "https:" + normalized
    parsed = urlparse(normalized)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        redirect_target = parse_qs(parsed.query).get("uddg", [""])[0]
        if redirect_target:
            return unquote(redirect_target)
    return normalized


def _search_with_serper(query, num_results=6):
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY is not configured")

    endpoint = os.getenv("SERPER_API_URL", "https://google.serper.dev/search")
    payload = {
        "q": query,
        "num": int(num_results),
        "gl": "us",
        "hl": "en",
    }
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    response = requests.post(endpoint, json=payload, headers=headers, timeout=20)
    response.raise_for_status()
    body = response.json()
    organic = body.get("organic", []) or []
    results = []
    for row in organic[:num_results]:
        results.append(
            {
                "url": row.get("link", ""),
                "title": row.get("title", ""),
                "description": row.get("snippet", ""),
            }
        )
    if not results:
        raise RuntimeError("Serper returned no organic results")
    return results


def _search_with_duckduckgo(query, num_results=6):
    response = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query, "kl": "us-en"},
        headers={"User-Agent": SEARCH_USER_AGENT},
        timeout=20,
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    results = []
    seen = set()
    for result in soup.select("div.result"):
        link = result.select_one("a.result__a")
        snippet = result.select_one(".result__snippet")
        if link is None:
            continue
        url = _normalize_duckduckgo_redirect_url(link.get("href", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": link.get_text(" ", strip=True),
                "description": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
        if len(results) >= num_results:
            break

    if not results:
        for link in soup.select("a.result-link"):
            url = _normalize_duckduckgo_redirect_url(link.get("href", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({"url": url, "title": link.get_text(" ", strip=True), "description": ""})
            if len(results) >= num_results:
                break

    if not results:
        raise RuntimeError("DuckDuckGo returned no results")
    return results


def _search_with_legacy_google(query, num_results=6):
    if google_search is None:
        raise RuntimeError("googlesearch dependency is unavailable")
    results = []
    for row in google_search(query, num_results=num_results, advanced=True, lang="en", region="US"):
        results.append(
            {
                "url": getattr(row, "url", ""),
                "title": getattr(row, "title", ""),
                "description": getattr(row, "description", ""),
            }
        )
    if not results:
        raise RuntimeError("Legacy Google search returned no results")
    return results


def search_google(query, num_results=6):
    query = str(query or "").strip()
    if not query:
        return "Error: query is required"
    try:
        num_results = int(num_results)
    except Exception:
        num_results = 6
    num_results = max(1, min(num_results, 10))

    preferred_provider = str(os.getenv("GPTPLUGINS_SEARCH_PROVIDER", "")).strip().lower()
    provider_orders = {
        "serper": ["serper", "duckduckgo", "google"],
        "duckduckgo": ["duckduckgo", "serper", "google"],
        "google": ["google", "serper", "duckduckgo"],
    }
    providers = provider_orders.get(preferred_provider, ["serper", "duckduckgo", "google"])
    logger.info("Searching web for query='%s' providers=%s", query, providers)

    errors = []
    provider_handlers = {
        "serper": _search_with_serper,
        "duckduckgo": _search_with_duckduckgo,
        "google": _search_with_legacy_google,
    }

    for provider in providers:
        handler = provider_handlers.get(provider)
        if handler is None:
            continue
        try:
            results = handler(query, num_results=num_results)
            return _format_search_results(query, provider, results)
        except Exception as e:
            logger.warning("Search provider '%s' failed for query '%s': %s", provider, query, e)
            errors.append(f"{provider}: {str(e)}")

    if errors:
        return "Error: web search failed for query '" + query + "'. " + " | ".join(errors)
    return "Error: web search failed for query '" + query + "'. No provider could return results."

def leftTruncate(text, length):
    encoded = encoding.encode(text)
    num = len(encoded)
    if num > length:
        return encoding.decode(encoded[num - length:])
    else:
        return text

def _playwright_scrape_subprocess(url):
    """Run Playwright in a subprocess to avoid conflicts with main application"""
    import subprocess
    import sys
    import json
    
    # Create a small Python script to run Playwright
    script = f'''
import sys
import json
try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("{url}", timeout=60000)
        page.wait_for_selector("body", timeout=30000)
        content = page.content()
        browser.close()
        print(json.dumps({{"success": True, "content": content}}))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
'''
    
    try:
        result = subprocess.run([sys.executable, '-c', script], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            data = json.loads(result.stdout.strip())
            if data.get("success"):
                return data.get("content", "")
            else:
                print(f"Playwright subprocess error: {data.get('error')}")
                return None
        else:
            print(f"Subprocess failed with return code {result.returncode}: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("Playwright subprocess timed out")
        return None
    except Exception as e:
        print(f"Error running Playwright subprocess: {e}")
        return None

def scrape_text(url, length):
    import re
    normalized_url = str(url or "").strip()
    if normalized_url.lower().startswith("file://"):
        return (
            "Error: file:// URLs are not supported by scrape_text. "
            "Use get_skill/get_master_doc for local guidance files instead."
        )
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    response = requests.get(normalized_url, verify=False, headers=headers)
    if not isinstance(length, int):
        length = 3000
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    # Check if we need to fall back to headless browser
    response_text = response.text
    # More comprehensive check: small content OR no links (indicating JS-heavy page)
    has_links = bool(re.search(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', response_text))
    content_too_small = len(response_text.strip()) < 100
    
    if content_too_small or not has_links:
        print(f"Falling back to headless browser for {normalized_url} (content_small: {content_too_small}, has_links: {has_links})")
        playwright_content = _playwright_scrape_subprocess(normalized_url)
        if playwright_content:
            response_text = playwright_content
            print(f"Successfully rendered page with headless browser, content length: {len(response_text)}")
        else:
            print("Playwright fallback failed, using original response")

    soup = BeautifulSoup(response_text, "html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = leftTruncate(text, length)
    return text

def encode_image(image_url):
    response = requests.get(image_url)
    #print(response.content)
    return base64.b64encode(response.content).decode('utf-8')
def generate_tools_from_api_calls(api_calls):
    import functools
    other_tools = []
    other_functions = {}

    for api_call in api_calls:
        # Use user-provided name or generate one
        function_name = api_call.get('name') or generate_function_name(api_call)
        # Use user-provided description or generate one
        function_description = api_call.get('description') or f"Call API endpoint {api_call['url']} with method {api_call['method']}"

        # Check for duplicate function names
        if function_name in other_functions:
            raise ValueError(f"Duplicate function name detected: {function_name}")

        # Define the parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        for param in api_call.get('params', []):
            parameters_schema["properties"][param] = {"type": "string"}
            parameters_schema["required"].append(param)

        # Create the tool definition
        tool = {
            "type": "function",
            "function": {
                "name": function_name,
                "description": function_description,
                "parameters": parameters_schema
            }
        }
        other_tools.append(tool)

        # Define the function that executes the API call
        def create_api_function(api_call):
            def api_function(arguments):
                import requests
                method = api_call['method']
                url = api_call['url']
                is_json = api_call.get('json', False)
                headers = api_call.get('headers', {}).copy()

                # Handle API key
                auth = api_call.get('auth', {})
                if auth:
                    api_key = auth.get('api_key', '')
                    placement = auth.get('placement', 'header')
                    if placement == 'header':
                        header_name = auth.get('header_name', 'Authorization')
                        format_str = auth.get('format', '{api_key}')
                        headers[header_name] = format_str.format(api_key=api_key)
                    elif placement == 'query':
                        param_name = auth.get('param_name', 'api_key')
                        arguments[param_name] = api_key
                    elif placement == 'body':
                        param_name = auth.get('param_name', 'api_key')
                        arguments[param_name] = api_key
                    else:
                        return f"Unsupported API key placement: {placement}"

                try:
                    if method.upper() == 'GET':
                        response = requests.get(url, params=arguments, headers=headers)
                    elif method.upper() == 'POST':
                        if is_json:
                            response = requests.post(url, json=arguments, headers=headers)
                        else:
                            response = requests.post(url, data=arguments, headers=headers)
                    elif method.upper() == 'PUT':
                        if is_json:
                            response = requests.put(url, json=arguments, headers=headers)
                        else:
                            response = requests.put(url, data=arguments, headers=headers)
                    elif method.upper() == 'DELETE':
                        response = requests.delete(url, params=arguments, headers=headers)
                    else:
                        return f"Unsupported HTTP method: {method}"

                    # Check for HTTP errors
                    response.raise_for_status()

                    # Return response text or json
                    try:
                        return response.json()
                    except ValueError:
                        return response.text
                except requests.exceptions.RequestException as e:
                    return f"An error occurred during the API call: {str(e)}"
            return api_function

        # Add the function to other_functions
        other_functions[function_name] = create_api_function(api_call)

    return other_tools, other_functions

def generate_function_name(api_call):
    from urllib.parse import urlparse
    parsed_url = urlparse(api_call['url'])
    path = parsed_url.path.strip('/').replace('/', '_').replace('-', '_').replace('.', '_')
    method = api_call['method'].lower()
    function_name = f"{method}_{path}"
    return function_name

class Assistant:
    def __init__(self, configs, name, instructions, model, assistant_id=None, thread_id=None, embedding_key=None,event_listener=None, openai_key=None, files=None, code_interpreter=False, retrieval=False, is_json=None, old_mode=False, max_tokens=None, bot_intro=None, get_thread=None, put_thread=None, save_memory=None, query_memory=None, max_messages=4, raw_mode=False, streaming=False, has_file=False, file_identifier=None, read_file=None, search_enabled=False, view_pages=False, search_window=1000, other_tools=None, other_functions={}, embedding_model=None, base_url=None, suggest_responses=False, api_calls=[], sources=None, initial_suggestions=None, mcp_servers=None, emit_tool_preamble=True, stop_check=None, async_tools=None, chat_completion_defaults=None, enable_context_compaction=False, context_budget_tokens=None, context_compact_threshold_ratio=0.82, context_compact_target_ratio=0.58, context_compact_keep_recent=18, tool_output_context_max_chars=1200, embedding_base_url=None, resolve_image_url=None):
        try:
            from openai import OpenAI
        except ImportError:
            OpenAI = None

        if OpenAI is None:
            raise ImportError("The OpenAI library is required to use this functionality. Please install it with `pip install GPTPlugins4All[openai]`.")
        if isinstance(configs, list):
            self.configs = configs
            self.multiple_configs = True
        else:
            self.configs = [configs]
            self.multiple_configs = False
        self.name = name
        self.instructions = instructions
        self.model = model
        self.event_listener = event_listener
        self.assistant_id = assistant_id
        self.embedding_model = embedding_model
        self.thread_id = thread_id
        self.old_mode = old_mode
        self.streaming = streaming
        self.has_file = has_file
        self.file_identifier = file_identifier
        self.read_file = read_file
        self.embedding_client = None
        self.search_enabled = search_enabled
        self.view_pages = view_pages
        self.search_window = search_window
        self.emit_tool_preamble = emit_tool_preamble
        self.stop_check = stop_check
        self.async_tools = async_tools or []
        self.chat_completion_defaults = copy.deepcopy(chat_completion_defaults or {})
        self.resolve_image_url = resolve_image_url
        self.enable_context_compaction = bool(enable_context_compaction)
        self.context_budget_tokens = int(context_budget_tokens or 0) if str(context_budget_tokens or "").strip() else 0
        self.context_compact_threshold_ratio = float(context_compact_threshold_ratio or 0.82)
        self.context_compact_target_ratio = float(context_compact_target_ratio or 0.58)
        self.context_compact_keep_recent = max(6, int(context_compact_keep_recent or 18))
        self.tool_output_context_max_chars = max(200, int(tool_output_context_max_chars or 1200))
        self._chat_payload_estimate_seq = 0
        self.other_tools = other_tools or []
        self.other_functions = other_functions or {}
        self.initial_suggestions = initial_suggestions
        self.chat_base_url = str(base_url or "").strip()
        self._chat_disallow_system_role = self._should_disallow_system_role(self.chat_base_url)
        self._openai_direct_client = None
        suggestions_str = ''
        if initial_suggestions is not None:
            suggestions_str = self._json_dumps_safe(initial_suggestions)

        # Generate tools and functions from api_calls
        more_tools, more_functions = generate_tools_from_api_calls(api_calls)
        self.other_tools += more_tools
        self.other_functions.update(more_functions)

        self.suggest_responses = suggest_responses
        if self.suggest_responses:
            self.instructions += "\nIn addition to the above, *always* give the user potential replies (eg quick-replies) to follow up with in this format: \n[\"response1\", \"response2\", \"response3\"]"
        if is_json is not None:
            self.is_json = is_json

        # Normalize empty-string keys to None so callers can "omit" keys cleanly.
        # This allows platform-key fallback behavior when OpenAI() is initialized
        # without an explicit api_key.
        if isinstance(openai_key, str) and not openai_key.strip():
            openai_key = None
        if isinstance(embedding_key, str) and not embedding_key.strip():
            embedding_key = None
        if isinstance(base_url, str) and not base_url.strip():
            base_url = None
        if isinstance(embedding_base_url, str) and not embedding_base_url.strip():
            embedding_base_url = None
        if embedding_base_url is None:
            embedding_base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL", "https://api.openai.com/v1")

        if openai_key is None:
            if base_url is None or base_url == '' or base_url == 'https://api.openai.com':
                self.openai_client = OpenAI()
            else:
                self.openai_client = OpenAI(base_url=base_url)
                self.embedding_client = OpenAI(api_key=embedding_key, base_url=embedding_base_url)
        else:
            if base_url is None or base_url == '' or base_url == 'https://api.openai.com':
                self.openai_client = OpenAI(api_key=openai_key)
            else:
                self.openai_client = OpenAI(api_key=openai_key, base_url=base_url)
                self.embedding_client = OpenAI(api_key=embedding_key, base_url=embedding_base_url)
        if old_mode:
            self.assistant = None
            self.thread = None
            self.old_mode = True
            self.raw_mode = raw_mode
            #if base_url is None or base_url == '' or base_url == 'https://api.openai.com':
            #    if get_thread is None:
            #        raise ValueError("get_thread must be provided if old_mode is True")
            #    if put_thread is None:
            #        raise ValueError("put_thread must be provided if old_mode is True")
            #    if max_tokens is None:
            #        raise ValueError("max_tokens must be provided if old_mode is True")
            self.save_memory = save_memory
            self.query_memory = query_memory
            self.max_messages = max_messages
            self.get_thread = get_thread
            self.put_thread = put_thread
            self.max_tokens = None
            self.max_completion_tokens = None
            if 'gpt-5' in model.lower():
                self.max_completion_tokens = max_tokens
            else:
                self.max_tokens = max_tokens
            pass
        else:
            self.assistant, self.thread = self.create_assistant_and_thread(files=files, code_interpreter=code_interpreter, retrieval=retrieval, bot_intro=bot_intro)
        
        # Initialize MCP clients
        self.mcp_servers = mcp_servers or {}
        self.mcp_sessions = {}
        self.mcp_tools = []
        self.mcp_functions = {}
        if self.mcp_servers:
            self._initialize_mcp_clients()

    def add_file(self, file):
        file = self.openai_client.create(
            file=open(file, 'rb'),
            purpose='assistants'
        )
        self.openai_client.beta.assistants.update(self.assistant_id, tool_resources={"code_interpreter": {"file_ids": [file.id]}})

    def _async_tool_system_hint(self, tool_names):
        if not tool_names:
            return None
        tool_list = ", ".join(sorted(set(tool_names)))
        return (
            f"Async tool(s) just started: {tool_list}. "
            "Do NOT call them again right now. "
            "Respond to the user by acknowledging the launch and what will happen next. "
            "If the tool output indicates a failure or missing data, explain the issue and ask a targeted follow-up."
        )
    
    def _build_chat_data(self, base_data):
        """Helper method to build chat completion data with correct token parameter"""
        data = copy.deepcopy(self.chat_completion_defaults)
        data.update(base_data)
        if 'gpt-5' in self.model.lower():
            if self.max_completion_tokens is not None:
                data["max_completion_tokens"] = self.max_completion_tokens
        else:
            if self.max_tokens is not None:
                data["max_tokens"] = self.max_tokens
        return data

    def _should_disallow_system_role(self, base_url):
        try:
            host = (urlparse(str(base_url or "")).netloc or "").lower()
        except Exception:
            host = ""
        return "minimaxi.chat" in host

    def _is_minimax_base_url(self):
        try:
            host = (urlparse(str(self.chat_base_url or "")).netloc or "").lower()
        except Exception:
            host = ""
        return "minimaxi.chat" in host

    def _payload_contains_image_inputs(self, payload):
        if not isinstance(payload, dict):
            return False
        for msg in (payload.get("messages") or []):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").strip().lower()
                if part_type in {"image_url", "input_image"}:
                    return True
        return False

    def _model_supports_image_inputs(self, model_name):
        model = str(model_name or "").strip().lower()
        if not model:
            model = str(self.model or "").strip().lower()
        if self._is_minimax_base_url():
            return False
        if "minimax" in model or "m2.5" in model:
            return False
        return True

    def _image_fallback_model(self):
        override = str(os.getenv("ASSISTANT_IMAGE_FALLBACK_MODEL", "") or "").strip()
        return override or "gpt-5.4-nano"

    def _get_openai_direct_client(self):
        if self._openai_direct_client is not None:
            return self._openai_direct_client
        try:
            from openai import OpenAI
        except Exception:
            return None
        try:
            api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip() or None
            if api_key:
                self._openai_direct_client = OpenAI(api_key=api_key)
            else:
                self._openai_direct_client = OpenAI()
        except Exception:
            self._openai_direct_client = None
        return self._openai_direct_client

    def _ensure_image_model_compatibility(self, payload):
        if not self._payload_contains_image_inputs(payload):
            return self.openai_client

        requested_model = str((payload or {}).get("model") or self.model or "").strip()
        if self._model_supports_image_inputs(requested_model):
            return self.openai_client

        fallback_model = self._image_fallback_model()
        if not fallback_model:
            return self.openai_client

        if requested_model.lower() != fallback_model.lower():
            logger.info(
                "Detected image input with non-image model '%s'; falling back to '%s'",
                requested_model,
                fallback_model,
            )

        payload["model"] = fallback_model
        if "gpt-5" in fallback_model.lower():
            if payload.get("max_completion_tokens") is None and payload.get("max_tokens") is not None:
                payload["max_completion_tokens"] = payload.get("max_tokens")
            payload.pop("max_tokens", None)

        if self._is_minimax_base_url():
            payload.pop("extra_body", None)
            direct_client = self._get_openai_direct_client()
            if direct_client is not None:
                return direct_client

        return self.openai_client

    def _system_prompt_as_user_content(self, content):
        text = self._content_to_text(content).strip()
        if not text:
            text = "(empty)"
        if text.startswith("[SYSTEM CONTEXT]"):
            return text
        return "[SYSTEM CONTEXT]\n" + text

    def _normalize_outbound_message(self, msg):
        if not isinstance(msg, dict):
            return None

        role = str(msg.get("role") or "").strip().lower()
        if role == "system":
            if self._chat_disallow_system_role:
                return {
                    "role": "user",
                    "content": self._system_prompt_as_user_content(msg.get("content")),
                }
            return {
                "role": "system",
                "content": self._content_to_text(msg.get("content")),
            }

        if role == "user":
            content = msg.get("content")
            if content is None:
                content = ""
            content = self._resolve_content_image_urls(content)
            return {"role": "user", "content": content}

        if role == "assistant":
            normalized = {"role": "assistant"}
            if msg.get("tool_calls") is not None:
                normalized["tool_calls"] = msg.get("tool_calls")
                normalized["content"] = self._resolve_content_image_urls(msg.get("content"))
            else:
                normalized["content"] = self._resolve_content_image_urls(
                    msg.get("content") if msg.get("content") is not None else ""
                )
            return normalized

        if role == "tool":
            normalized = {
                "role": "tool",
                "tool_call_id": msg.get("tool_call_id"),
                "content": msg.get("content") if msg.get("content") is not None else "",
            }
            if msg.get("name"):
                normalized["name"] = msg.get("name")
            return normalized

        fallback_content = self._content_to_text(msg.get("content"))
        return {"role": "user", "content": fallback_content}

    def _prepare_chat_payload(self, data):
        payload = copy.deepcopy(data or {})
        payload_messages = payload.get("messages") or []
        sanitized_messages = []
        for msg in payload_messages:
            normalized = self._normalize_outbound_message(msg)
            if normalized is None:
                continue
            if normalized.get("role") == "tool" and not normalized.get("tool_call_id"):
                continue
            sanitized_messages.append(normalized)
        payload["messages"] = sanitized_messages
        payload["tools"] = self._normalize_tool_schemas(payload.get("tools"))
        payload["tools"] = self._prune_tools_for_provider(
            payload.get("tools"),
            payload.get("messages") or [],
        )

        if not payload.get("tools"):
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
        elif payload.get("tool_choice") is None:
            payload.pop("tool_choice", None)
        return payload

    def _max_tools_for_provider(self):
        try:
            model_name = str(self.model or "").strip().lower()
        except Exception:
            model_name = ""
        if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
            return OPENAI_CHAT_TOOL_LIMIT
        if not self.chat_base_url:
            return OPENAI_CHAT_TOOL_LIMIT
        try:
            host = (urlparse(str(self.chat_base_url or "")).netloc or "").lower()
        except Exception:
            host = ""
        if host.endswith("openai.com") or host == "api.openai.com":
            return OPENAI_CHAT_TOOL_LIMIT
        return 0

    def _message_text_for_tool_selection(self, messages):
        if not isinstance(messages, list):
            return ""
        chunks = []
        for msg in reversed(messages[-8:]):
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"user", "assistant", "system"}:
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
                if text:
                    chunks.append(text)
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get("type") or "").strip().lower()
                        if item_type in {"text", "input_text"}:
                            text = str(item.get("text") or "").strip()
                            if text:
                                parts.append(text)
                            continue
                        if item_type in {"image_url", "input_image"}:
                            parts.append("image")
                            continue
                        if item_type in {"video_url", "input_video"}:
                            parts.append("video")
                            continue
                        text = str(item.get("content") or "").strip()
                        if text:
                            parts.append(text)
                    else:
                        text = str(item or "").strip()
                        if text:
                            parts.append(text)
                if parts:
                    chunks.append(" ".join(parts))
            if len(chunks) >= 3:
                break
        return "\n".join(reversed(chunks)).strip()

    def _tool_selection_terms(self, message_text):
        raw = str(message_text or "").lower()
        tokens = set(re.findall(r"[a-z0-9_]{3,}", raw))
        phrase_tokens = set()
        for token in list(tokens):
            if "_" in token:
                phrase_tokens.update([part for part in token.split("_") if len(part) >= 3])
        return tokens | phrase_tokens

    def _tool_selection_score(self, tool, message_text, selection_terms, original_index):
        if not isinstance(tool, dict):
            return (-100000, -original_index)
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = str(fn.get("name") or tool.get("type") or "").strip().lower()
        description = str(fn.get("description") or "").strip().lower()
        haystack = f"{name} {description}".strip()
        haystack_terms = self._tool_selection_terms(haystack)

        score = 0
        if name:
            if name in message_text:
                score += 300
            name_parts = [part for part in re.split(r"[^a-z0-9]+", name) if len(part) >= 3]
            if name_parts:
                exact_name_overlap = sum(1 for part in name_parts if part in selection_terms)
                score += exact_name_overlap * 45
        overlap = len(selection_terms & haystack_terms)
        score += overlap * 12

        search_terms = {"search", "find", "lookup", "look", "google", "web", "website", "site", "browse"}
        media_terms = {"video", "image", "stills", "still", "frame", "frames", "media", "canvas", "ffmpeg", "daytona"}
        workspace_terms = {"workspace", "file", "files", "path", "upload", "download", "artifact"}
        connector_terms = {"api", "connector", "endpoint", "webhook", "hubspot", "salesforce"}
        linkedin_terms = {"linkedin", "comment", "post", "reply", "connect", "pipeline", "sequence", "campaign"}

        if name in {"search_google", "scrape_text"} and selection_terms & search_terms:
            score += 120
        if name.startswith("operator_") and selection_terms & workspace_terms:
            score += 80
        if name.startswith("operator_") and selection_terms & media_terms:
            score += 140
        if ("connector" in name or "api" in name) and selection_terms & connector_terms:
            score += 90
        if ("linkedin" in haystack or "campaign" in haystack or "sequence" in haystack) and selection_terms & linkedin_terms:
            score += 90

        return (score, -original_index)

    def _prune_tools_for_provider(self, tools, messages):
        if not isinstance(tools, list):
            return tools
        max_tools = self._max_tools_for_provider()
        if max_tools <= 0 or len(tools) <= max_tools:
            return tools

        deduped = []
        seen_names = set()
        for tool in tools:
            if not isinstance(tool, dict):
                deduped.append(tool)
                continue
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = str(fn.get("name") or "").strip()
            key = name or self._json_dumps_safe(tool)
            if key in seen_names:
                continue
            seen_names.add(key)
            deduped.append(tool)

        if len(deduped) <= max_tools:
            return deduped

        message_text = self._message_text_for_tool_selection(messages).lower()
        selection_terms = self._tool_selection_terms(message_text)
        ranked = []
        for index, tool in enumerate(deduped):
            ranked.append((self._tool_selection_score(tool, message_text, selection_terms, index), index, tool))
        ranked.sort(key=lambda row: (row[0][0], row[0][1]), reverse=True)
        selected = sorted(ranked[:max_tools], key=lambda row: row[1])
        selected_tools = [row[2] for row in selected]

        try:
            dropped_names = []
            for _score, _index, tool in ranked[max_tools:]:
                fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
                dropped_names.append(str(fn.get("name") or tool.get("type") or "unknown"))
            logger.info(
                "Assistant pruned tools for provider limit: kept=%s dropped=%s model=%s last_user_terms=%s",
                len(selected_tools),
                len(ranked) - len(selected_tools),
                self.model,
                sorted(list(selection_terms))[:20],
            )
            if dropped_names:
                logger.debug("Assistant dropped tool names: %s", dropped_names[:50])
        except Exception:
            pass

        return selected_tools

    def _tool_call_signature(self, tool_name, tool_arguments):
        normalized_name = str(tool_name or "").strip()
        raw_args = tool_arguments
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except Exception:
                raw_args = str(tool_arguments or "").strip()
        try:
            normalized_args = json.dumps(raw_args, sort_keys=True, default=str)
        except Exception:
            normalized_args = str(raw_args)
        return f"{normalized_name}::{normalized_args}"

    def _tool_result_success(self, result):
        if isinstance(result, dict):
            if result.get("duplicate_call_blocked"):
                return False
            if "success" in result:
                return bool(result.get("success"))
            if result.get("error"):
                return False
        return True

    def _duplicate_tool_output(self, tool_name, tool_arguments, prior_output):
        return {
            "duplicate_call_blocked": True,
            "tool_name": str(tool_name or "").strip(),
            "tool_arguments": tool_arguments,
            "message": "This exact tool call already succeeded in this turn. Reuse the prior result and answer the user instead of calling it again.",
            "prior_output_preview": self._normalize_text_for_context(
                self._json_dumps_safe(prior_output),
                max_chars=min(600, self.tool_output_context_max_chars),
            ),
        }

    def _normalize_tool_schema_fragment(self, schema_fragment):
        if isinstance(schema_fragment, list):
            return [self._normalize_tool_schema_fragment(item) for item in schema_fragment]
        if not isinstance(schema_fragment, dict):
            return schema_fragment

        normalized = {
            key: self._normalize_tool_schema_fragment(value)
            for key, value in schema_fragment.items()
        }
        type_value = normalized.get("type")
        has_array_type = (
            type_value == "array"
            or (
                isinstance(type_value, list)
                and any(str(t or "").strip().lower() == "array" for t in type_value)
            )
        )
        if has_array_type and "items" not in normalized:
            normalized["items"] = {"type": "string"}
        if normalized.get("enum") == []:
            normalized.pop("enum", None)
        return normalized

    def _normalize_tool_schemas(self, tools):
        if not isinstance(tools, list):
            return tools
        normalized_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                normalized_tools.append(tool)
                continue
            tool_copy = copy.deepcopy(tool)
            fn = tool_copy.get("function")
            if isinstance(fn, dict):
                params = fn.get("parameters")
                if isinstance(params, dict):
                    fn["parameters"] = self._normalize_tool_schema_fragment(params)
            normalized_tools.append(tool_copy)
        return normalized_tools

    def _normalize_text_for_context(self, text, max_chars=500, collapse_whitespace=True):
        raw = str(text or "")
        if collapse_whitespace:
            raw = re.sub(r"\s+", " ", raw).strip()
        else:
            raw = raw.strip()
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "... [truncated]"
        return raw

    def _json_dumps_safe(self, value, *, indent=None):
        try:
            return json.dumps(value, default=str, indent=indent)
        except Exception:
            try:
                return json.dumps(str(value), indent=indent)
            except Exception:
                return "\"<serialization_failed>\""

    def _coerce_reasoning_text(self, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [self._coerce_reasoning_text(v) for v in value]
            return "\n".join([p for p in parts if p]).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "reasoning"):
                v = value.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            try:
                return json.dumps(value, default=str)
            except Exception:
                return str(value).strip()
        return str(value).strip()

    def _extract_reasoning_from_obj(self, obj):
        if obj is None:
            return ""
        parts = []
        for attr in ("reasoning", "reasoning_content", "reasoning_text"):
            try:
                value = getattr(obj, attr, None)
            except Exception:
                value = None
            text = self._coerce_reasoning_text(value)
            if text:
                parts.append(text)
        try:
            details = getattr(obj, "reasoning_details", None)
        except Exception:
            details = None
        details_text = self._coerce_reasoning_text(details)
        if details_text:
            parts.append(details_text)
        merged = "\n".join([p for p in parts if p]).strip()
        return merged

    def _strip_think_blocks(self, text):
        raw = str(text or "")
        if not raw:
            return "", ""
        pattern = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
        extracted = []

        def _capture(match):
            body = (match.group(1) or "").strip()
            if body:
                extracted.append(body)
            return ""

        cleaned = pattern.sub(_capture, raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        reasoning = "\n\n".join([x for x in extracted if x]).strip()
        return cleaned, reasoning

    def _strip_legacy_tool_call_blocks(self, text):
        raw = str(text or "")
        if not raw:
            return "", []
        pattern = re.compile(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", re.IGNORECASE | re.DOTALL)
        blocks = []

        def _capture(match):
            body = str(match.group(1) or "").strip()
            if body:
                blocks.append(body)
            return ""

        cleaned = pattern.sub(_capture, raw)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, blocks

    def _parse_legacy_tool_value(self, raw_value):
        value = str(raw_value or "").strip().rstrip(",")
        if not value:
            return True
        if value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "null":
            return None
        try:
            if re.match(r"^-?\d+$", value):
                return int(value)
            if re.match(r"^-?\d+\.\d+$", value):
                return float(value)
        except Exception:
            pass
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value)
            except Exception:
                continue
        return value

    def _parse_legacy_tool_calls(self, text):
        _cleaned, blocks = self._strip_legacy_tool_call_blocks(text)
        parsed_calls = []
        for idx, block in enumerate(blocks):
            tool_match = re.search(r'tool\s*=>\s*["\']([^"\']+)["\']', block, re.IGNORECASE)
            if not tool_match:
                tool_match = re.search(r'tool\s*:\s*["\']([^"\']+)["\']', block, re.IGNORECASE)
            tool_name = str((tool_match.group(1) if tool_match else "") or "").strip()
            if not tool_name:
                continue

            args = {}
            args_match = re.search(r"args\s*=>\s*\{(.*)\}\s*$", block, re.IGNORECASE | re.DOTALL)
            if not args_match:
                args_match = re.search(r"args\s*:\s*\{(.*)\}\s*$", block, re.IGNORECASE | re.DOTALL)
            args_body = str((args_match.group(1) if args_match else "") or "")
            if args_body:
                for raw_line in args_body.splitlines():
                    line = str(raw_line or "").strip()
                    if not line or not line.startswith("--"):
                        continue
                    match = re.match(r"--([a-zA-Z0-9_-]+)(?:\s+(.*))?$", line)
                    if not match:
                        continue
                    key = str(match.group(1) or "").strip().replace("-", "_")
                    if not key:
                        continue
                    args[key] = self._parse_legacy_tool_value(match.group(2) or "")

            parsed_calls.append(
                {
                    "id": f"legacy_call_{idx}",
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=True),
                }
            )
        return parsed_calls

    def _merge_reasoning_text(self, *parts):
        merged = []
        seen = set()
        for part in parts:
            txt = self._coerce_reasoning_text(part)
            if not txt:
                continue
            key = txt.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(key)
        return "\n\n".join(merged).strip()

    def _emit_reasoning_event(self, reasoning_text):
        txt = self._coerce_reasoning_text(reasoning_text)
        if not txt or not self.event_listener:
            return
        try:
            self.event_listener({"type": "assistant_reasoning", "content": txt})
        except Exception:
            pass

    def _consume_stream_think_chunk(self, chunk, state):
        text = str(chunk or "")
        if not text:
            return "", ""
        pending = str(state.get("pending") or "") + text
        in_think = bool(state.get("in_think"))
        visible_out = []
        reasoning_out = []

        while True:
            if in_think:
                end_idx = pending.find("</think>")
                if end_idx >= 0:
                    reasoning_out.append(pending[:end_idx])
                    pending = pending[end_idx + len("</think>"):]
                    in_think = False
                    continue
                safe_len = max(0, len(pending) - (len("</think>") - 1))
                if safe_len > 0:
                    reasoning_out.append(pending[:safe_len])
                    pending = pending[safe_len:]
                break
            else:
                start_idx = pending.find("<think>")
                if start_idx >= 0:
                    visible_out.append(pending[:start_idx])
                    pending = pending[start_idx + len("<think>"):]
                    in_think = True
                    continue
                safe_len = max(0, len(pending) - (len("<think>") - 1))
                if safe_len > 0:
                    visible_out.append(pending[:safe_len])
                    pending = pending[safe_len:]
                break

        state["pending"] = pending
        state["in_think"] = in_think
        return "".join(visible_out), "".join(reasoning_out)

    def _consume_stream_legacy_tool_call_chunk(self, chunk, state):
        text = str(chunk or "")
        if not text:
            return ""
        pending = str(state.get("pending") or "") + text
        in_tool_call = bool(state.get("in_tool_call"))
        visible_out = []
        start_tag = "[TOOL_CALL]"
        end_tag = "[/TOOL_CALL]"

        while True:
            if in_tool_call:
                end_idx = pending.find(end_tag)
                if end_idx >= 0:
                    pending = pending[end_idx + len(end_tag):]
                    in_tool_call = False
                    continue
                safe_len = max(0, len(pending) - (len(end_tag) - 1))
                if safe_len > 0:
                    pending = pending[safe_len:]
                break
            start_idx = pending.find(start_tag)
            if start_idx >= 0:
                visible_out.append(pending[:start_idx])
                pending = pending[start_idx + len(start_tag):]
                in_tool_call = True
                continue
            safe_len = max(0, len(pending) - (len(start_tag) - 1))
            if safe_len > 0:
                visible_out.append(pending[:safe_len])
                pending = pending[safe_len:]
            break

        state["pending"] = pending
        state["in_tool_call"] = in_tool_call
        return "".join(visible_out)

    def _flush_stream_legacy_tool_call_state(self, state):
        pending = str(state.get("pending") or "")
        in_tool_call = bool(state.get("in_tool_call"))
        state["pending"] = ""
        state["in_tool_call"] = False
        if not pending or in_tool_call:
            return ""
        return pending

    def _flush_stream_think_state(self, state):
        pending = str(state.get("pending") or "")
        in_think = bool(state.get("in_think"))
        state["pending"] = ""
        state["in_think"] = False
        if not pending:
            return "", ""
        if in_think:
            return "", pending
        return pending, ""

    def _content_to_text(self, content):
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item.get("text"))
                    elif isinstance(item.get("content"), str):
                        parts.append(item.get("content"))
            return "\n".join([p for p in parts if p])
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content.get("text")
            if isinstance(content.get("content"), str):
                return content.get("content")
            try:
                return json.dumps(content, default=str)
            except Exception:
                return str(content)
        return str(content)

    def _resolve_image_url_value(self, url, source_url=None):
        candidate = str(source_url or url or "").strip()
        if not candidate:
            return str(url or "").strip()
        resolver = self.resolve_image_url
        if callable(resolver):
            try:
                resolved = resolver(candidate)
                resolved_str = str(resolved or "").strip()
                if resolved_str:
                    return resolved_str
            except Exception:
                pass
        return str(url or candidate).strip()

    def _extract_tool_generated_image_urls(self, tool_outputs):
        urls = []
        seen = set()

        def _add(candidate):
            raw = str(candidate or "").strip()
            if not raw:
                return
            resolved = self._resolve_image_url_value(raw, raw)
            resolved = str(resolved or "").strip()
            if not resolved:
                return
            low = resolved.lower()
            if not (low.startswith("https://") or low.startswith("http://")):
                return
            if resolved in seen:
                return
            seen.add(resolved)
            urls.append(resolved)

        for row in tool_outputs or []:
            payload = row.get("output") if isinstance(row, dict) else None
            parsed = None
            if isinstance(payload, dict):
                parsed = payload
            elif isinstance(payload, str):
                try:
                    parsed = json.loads(payload)
                except Exception:
                    parsed = None
            if not isinstance(parsed, dict):
                continue

            generated = parsed.get("generated_image_urls")
            if isinstance(generated, list):
                for item in generated:
                    _add(item)

            artifact = parsed.get("workspace_artifact")
            if isinstance(artifact, dict):
                _add(artifact.get("model_image_url"))
                _add(artifact.get("image_url"))
                storage = artifact.get("storage")
                if isinstance(storage, dict):
                    _add(storage.get("uri"))
                    _add(storage.get("path"))

            artifacts = parsed.get("workspace_artifacts")
            if isinstance(artifacts, list):
                for item in artifacts:
                    if not isinstance(item, dict):
                        continue
                    _add(item.get("model_image_url"))
                    _add(item.get("image_url"))
                    storage = item.get("storage")
                    if isinstance(storage, dict):
                        _add(storage.get("uri"))
                        _add(storage.get("path"))

        return urls[:4]

    def _tool_image_context_message(self, image_urls):
        rows = [str(url or "").strip() for url in (image_urls or []) if str(url or "").strip()]
        if not rows:
            return None
        content = [
            {
                "type": "text",
                "text": "Connector-generated image(s) from the latest tool call. Use these for visual analysis and iteration if helpful.",
            }
        ]
        for url in rows[:4]:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return {"role": "user", "content": content}

    def _normalize_image_input(self, image_input):
        if isinstance(image_input, dict):
            source_url = str(
                image_input.get("source_url")
                or image_input.get("s3_uri")
                or image_input.get("agent_image_path")
                or image_input.get("public_url")
                or image_input.get("url")
                or image_input.get("image_url")
                or ""
            ).strip()
            display_url = str(
                image_input.get("url")
                or image_input.get("access_url")
                or image_input.get("image_url")
                or source_url
                or ""
            ).strip()
            return display_url, source_url
        raw = str(image_input or "").strip()
        return raw, raw

    def _resolve_content_image_urls(self, content):
        if isinstance(content, list):
            out = []
            for part in content:
                if not isinstance(part, dict):
                    out.append(part)
                    continue
                next_part = dict(part)
                if str(next_part.get("type") or "").strip().lower() == "image_url":
                    image_url = next_part.get("image_url")
                    if isinstance(image_url, dict):
                        image_payload = dict(image_url)
                        original_url = str(image_payload.get("url") or "").strip()
                        source_url = str(next_part.get("source_url") or "").strip()
                        resolved_url = self._resolve_image_url_value(original_url, source_url=source_url)
                        if resolved_url:
                            image_payload["url"] = resolved_url
                        next_part["image_url"] = image_payload
                    next_part.pop("source_url", None)
                out.append(next_part)
            return out
        return content

    def _estimate_messages_tokens(self, messages, prefix_text=""):
        try:
            enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            enc = encoding
        total = 4
        if prefix_text:
            try:
                total += len(enc.encode(str(prefix_text)))
            except Exception:
                total += int(len(str(prefix_text)) / 4)
        for msg in messages or []:
            try:
                total += 4
                total += len(enc.encode(str((msg or {}).get("role") or "")))
                total += len(enc.encode(self._content_to_text((msg or {}).get("content"))))
            except Exception:
                total += int(len(self._content_to_text((msg or {}).get("content"))) / 4) + 8
        return total

    def _extract_compaction_lists(self, messages):
        state_summary = []
        open_loops = []
        decision_log = []
        sticky_rules = []
        apollo_phone_persist_rule = (
            "If Apollo enrichment returns phone numbers, immediately persist them to the person/contact record before any next step."
        )
        saw_apollo_phone_context = False
        for msg in (messages or [])[-200:]:
            role = str((msg or {}).get("role") or "unknown").strip().lower()
            text = self._normalize_text_for_context(self._content_to_text((msg or {}).get("content")), max_chars=360)
            if not text:
                continue
            if "[context_compaction_v1]" in text.lower():
                continue
            if role in {"user", "assistant"}:
                state_summary.append(f"{role}: {text}")
            lower = text.lower()
            if (
                "operator_sync_apollo_people_enrichment" in lower
                or ("apollo" in lower and ("phone" in lower or "mobile" in lower or "direct dial" in lower))
            ):
                saw_apollo_phone_context = True
            if "?" in text or "todo" in lower or "follow up" in lower or "next step" in lower:
                open_loops.append(text)
            if role == "assistant" and ("i will" in lower or "we will" in lower or "plan:" in lower or "next," in lower):
                decision_log.append(text)
        if saw_apollo_phone_context:
            sticky_rules.append(apollo_phone_persist_rule)
        return (
            state_summary[:24],
            open_loops[:16],
            decision_log[:16],
            sticky_rules[:8],
        )

    def _render_compaction_message(self, state_summary, open_loops, decision_log, sticky_rules, removed_count):
        lines = [
            "[context_compaction_v1]",
            f"compacted_messages: {int(removed_count)}",
            "state_summary:",
        ]
        if state_summary:
            lines.extend([f"- {x}" for x in state_summary[:16]])
        else:
            lines.append("- (none)")
        lines.append("open_loops:")
        if open_loops:
            lines.extend([f"- {x}" for x in open_loops[:12]])
        else:
            lines.append("- (none)")
        lines.append("decision_log:")
        if decision_log:
            lines.extend([f"- {x}" for x in decision_log[:12]])
        else:
            lines.append("- (none)")
        lines.append("sticky_rules:")
        if sticky_rules:
            lines.extend([f"- {x}" for x in sticky_rules[:8]])
        else:
            lines.append("- (none)")
        lines.append("[/context_compaction_v1]")
        return "\n".join(lines)

    def _emit_compaction_event(self, meta):
        if not self.event_listener:
            return
        try:
            self.event_listener({
                "type": "context_compaction",
                "before_tokens": meta.get("before_tokens"),
                "after_tokens": meta.get("after_tokens"),
                "removed_messages": meta.get("removed_messages"),
                "reason": meta.get("reason"),
                "input_tokens": meta.get("input_tokens"),
                "message_tokens": meta.get("message_tokens"),
                "tool_schema_tokens": meta.get("tool_schema_tokens"),
            })
        except Exception:
            pass

    def _maybe_compact_messages(self, messages, additional_context=""):
        if not self.enable_context_compaction:
            return messages, None
        if not isinstance(messages, list) or len(messages) < (self.context_compact_keep_recent + 2):
            return messages, None
        budget = int(self.context_budget_tokens or 0)
        if budget <= 0:
            return messages, None
        threshold = max(1, int(budget * self.context_compact_threshold_ratio))
        target = max(1, int(budget * self.context_compact_target_ratio))
        if target >= threshold:
            target = max(1, threshold - 1)

        prefix = f"{self.instructions}\n{additional_context or ''}"
        before_tokens = self._estimate_messages_tokens(messages, prefix_text=prefix)
        if before_tokens <= threshold:
            return messages, None

        keep_recent = int(self.context_compact_keep_recent)
        existing_summary = None
        body = list(messages)
        if (
            body
            and isinstance(body[0], dict)
            and str(body[0].get("role") or "").lower() == "system"
            and "[context_compaction_v1]" in str(body[0].get("content") or "")
        ):
            existing_summary = body[0]
            body = body[1:]

        sticky_system = []
        compactable = []
        for msg in body:
            role = str((msg or {}).get("role") or "").lower()
            content = str((msg or {}).get("content") or "")
            if role == "system":
                if "Tool outputs from most recent attempt" not in content and "[context_compaction_v1]" not in content:
                    sticky_system.append(msg)
                    continue
            compactable.append(msg)

        if len(compactable) <= keep_recent:
            return messages, None

        while keep_recent >= 6:
            preserved = compactable[-keep_recent:]
            removed = compactable[:-keep_recent]
            pool = list(removed)
            if existing_summary is not None:
                pool = [existing_summary] + pool
            state_summary, open_loops, decision_log, sticky_rules = self._extract_compaction_lists(pool)
            compact_msg = {
                "role": "system",
                "content": self._render_compaction_message(
                    state_summary=state_summary,
                    open_loops=open_loops,
                    decision_log=decision_log,
                    sticky_rules=sticky_rules,
                    removed_count=len(removed),
                ),
                "timestamp": datetime.now().isoformat(),
            }
            candidate = sticky_system + [compact_msg] + preserved
            after_tokens = self._estimate_messages_tokens(candidate, prefix_text=prefix)
            if after_tokens <= target or keep_recent == 6:
                meta = {
                    "before_tokens": before_tokens,
                    "after_tokens": after_tokens,
                    "removed_messages": len(removed),
                }
                return candidate, meta
            keep_recent = max(6, keep_recent - 4)

        return messages, None

    def _drop_tool_result_key(self, key):
        normalized = str(key or "").strip().lower()
        if not normalized:
            return True
        if normalized in {
            "_run_memory",
            "_run_memory_hint",
            "tenant_id",
            "content_hash",
            "raw_data",
            "raw_response",
            "raw_html",
            "html",
            "debug",
            "trace",
        }:
            return True
        if normalized in {"created_at", "updated_at", "processed_at", "last_manual_targeting_seen_at"}:
            return True
        if normalized.startswith("last_") and normalized.endswith("_at"):
            return True
        return False

    def _compact_person_payload(self, value):
        if not isinstance(value, dict):
            return value
        keep = (
            "person_id",
            "name",
            "title",
            "headline",
            "company_id",
            "company_name",
            "email",
            "phone_number",
            "linkedin_url",
            "tracking_stage",
        )
        return {
            key: self._sanitize_tool_result_value(value.get(key), key=key, depth=1)
            for key in keep
            if value.get(key) not in (None, "", [], {})
        }

    def _compact_company_payload(self, value):
        if not isinstance(value, dict):
            return value
        keep = (
            "company_id",
            "id",
            "name",
            "company_name",
            "website",
            "domain",
            "linkedin_url",
            "profile_url",
            "industry",
            "location",
            "description",
        )
        return {
            key: self._sanitize_tool_result_value(value.get(key), key=key, depth=1)
            for key in keep
            if value.get(key) not in (None, "", [], {})
        }

    def _sanitize_tool_result_value(self, value, *, key="", depth=0):
        normalized_key = str(key or "").strip().lower()
        if self._drop_tool_result_key(normalized_key):
            return None
        if depth > 5:
            return self._normalize_text_for_context(value, max_chars=240, collapse_whitespace=True)
        if isinstance(value, str):
            limit = 900
            if normalized_key in {"description", "summary", "result_summary", "message", "error", "reason"}:
                limit = max(limit, self.tool_output_context_max_chars)
            if normalized_key in {"email", "phone_number", "linkedin_url", "website", "url", "profile_url"}:
                limit = 400
            return self._normalize_text_for_context(value, max_chars=limit, collapse_whitespace=True)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            cap = 8
            if normalized_key in {"people", "persons", "companies", "signals", "items", "matches"}:
                cap = 10
            rows = []
            for item in value[:cap]:
                cleaned = self._sanitize_tool_result_value(item, key=normalized_key, depth=depth + 1)
                if cleaned not in (None, "", [], {}):
                    rows.append(cleaned)
            if len(value) > cap:
                rows.append({"truncated_count": len(value) - cap})
            return rows
        if isinstance(value, dict):
            if normalized_key in {"person", "people", "persons"} and {"person_id", "name"} & set(value.keys()):
                return self._compact_person_payload(value)
            if normalized_key in {"company", "companies"} and ({"company_id", "name", "company_name"} & set(value.keys())):
                return self._compact_company_payload(value)
            out = {}
            priority_keys = (
                "success",
                "error",
                "message",
                "reason",
                "status",
                "run_id",
                "workflow_run_id",
                "account_id",
                "company_id",
                "company_name",
                "person_id",
                "dossier_id",
                "sequence_id",
                "campaign_id",
                "workspace_path",
                "path",
                "count",
                "created",
                "updated",
                "launched",
                "skipped",
                "person",
                "people",
                "company",
                "companies",
                "signals",
                "items",
                "matches",
                "selected_targets",
                "rate_limits",
            )
            ordered_keys = [k for k in priority_keys if k in value]
            ordered_keys.extend([k for k in value.keys() if k not in ordered_keys])
            for child_key in ordered_keys:
                if len(out) >= 28:
                    out["truncated_keys"] = max(0, len(value) - len(out))
                    break
                if self._drop_tool_result_key(child_key):
                    continue
                cleaned = self._sanitize_tool_result_value(value.get(child_key), key=child_key, depth=depth + 1)
                if cleaned not in (None, "", [], {}):
                    out[str(child_key)] = cleaned
            return out
        return self._normalize_text_for_context(value, max_chars=500, collapse_whitespace=True)

    def _parse_tool_result_for_model(self, output):
        raw_output = (output or {}).get("output")
        parsed = raw_output
        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
            except Exception:
                parsed = raw_output
        return self._sanitize_tool_result_value(parsed, key="result", depth=0)

    def _model_visible_tool_outputs(self, compact_outputs):
        rows = []
        for output in compact_outputs or []:
            result = (output or {}).get("model_result")
            if result in (None, "", [], {}):
                result = (output or {}).get("output")
            rows.append(
                {
                    "tool": str((output or {}).get("tool_name") or "").strip(),
                    "result": result,
                }
            )
        return rows

    def _compact_tool_outputs_for_context(self, tool_outputs):
        compacted = []
        for output in (tool_outputs or []):
            row = dict(output or {})
            tool_name = str(row.get("tool_name") or "").strip().lower()
            max_chars = self.tool_output_context_max_chars
            # Apollo enrichment payloads can include contact fields near the tail of
            # large JSON objects; use a larger context budget so email/phone values
            # are still available to the model on the follow-up turn.
            if "apollo" in tool_name and ("enrich" in tool_name or "person" in tool_name or "contact" in tool_name):
                max_chars = max(max_chars, 12000)
            model_result = self._parse_tool_result_for_model(row)
            row["output"] = self._normalize_text_for_context(
                self._json_dumps_safe(model_result),
                max_chars=max_chars,
                collapse_whitespace=False,
            )
            row["model_result"] = model_result
            compacted.append(row)
        return compacted

    def _tool_output_followup_hint(self, compact_outputs, *, force_no_more_tools=False, include_outputs=True):
        if include_outputs:
            message = (
                "Tool outputs from most recent attempt: "
                + self._json_dumps_safe(self._model_visible_tool_outputs(compact_outputs))
                + "\n"
            )
        else:
            message = "Use the preceding tool result messages as the source of truth. "
        message += (
            "If the tool output indicates success and already contains the requested result, do not call more tools. "
            + "Answer the user using the latest tool result. Only retry a tool if the output indicates failure or missing required data."
        )
        if force_no_more_tools:
            message += " Tool-call budget is exhausted for this turn. Do not call any more tools."
        return message

    def _is_openrouter_base_url(self):
        try:
            host = (urlparse(str(self.chat_base_url or "")).netloc or "").lower()
        except Exception:
            host = ""
        return "openrouter.ai" in host

    def _safe_int_env(self, name, default):
        try:
            return int(str(os.getenv(name, str(default))).strip())
        except Exception:
            return int(default)

    def _max_tool_rounds(self):
        explicit = self._safe_int_env("ASSISTANT_MAX_TOOL_ROUNDS", 0)
        if explicit > 0:
            return max(1, explicit)
        if self._is_minimax_base_url():
            return max(1, self._safe_int_env("ASSISTANT_MAX_TOOL_ROUNDS_MINIMAX", 80))
        return max(1, self._safe_int_env("ASSISTANT_MAX_TOOL_ROUNDS_DEFAULT", 6))

    def _context_limit_tokens(self):
        explicit = self._safe_int_env("ASSISTANT_CONTEXT_LIMIT_TOKENS", 0)
        if explicit > 0:
            return explicit
        if self._is_openrouter_base_url():
            return self._safe_int_env("ASSISTANT_OPENROUTER_CONTEXT_LIMIT_TOKENS", 204800)
        return 0

    def _min_output_tokens_floor(self):
        return max(256, self._safe_int_env("ASSISTANT_MIN_COMPLETION_TOKENS", 60000))

    def _context_retry_margin_tokens(self):
        return max(64, self._safe_int_env("ASSISTANT_CONTEXT_RETRY_MARGIN_TOKENS", 512))

    def _context_error_max_retries(self):
        return max(1, self._safe_int_env("ASSISTANT_CONTEXT_ERROR_MAX_RETRIES", 4))

    def _is_context_length_error(self, err_text):
        raw = str(err_text or "").lower()
        return ("context length" in raw and "token" in raw) or ("maximum context length" in raw)

    def _extract_context_error_meta(self, err_text):
        raw = str(err_text or "")
        if not raw:
            return None

        detailed = re.search(
            r"maximum context length is\s*([\d,]+)\s*tokens.*?requested about\s*([\d,]+)\s*tokens\s*\(\s*([\d,]+)\s*of text input,\s*([\d,]+)\s*of tool input,\s*([\d,]+)\s*in the output\s*\)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if detailed:
            limit = int(detailed.group(1).replace(",", ""))
            requested = int(detailed.group(2).replace(",", ""))
            text_input = int(detailed.group(3).replace(",", ""))
            tool_input = int(detailed.group(4).replace(",", ""))
            output = int(detailed.group(5).replace(",", ""))
            return {
                "context_limit": limit,
                "requested_tokens": requested,
                "text_input_tokens": text_input,
                "tool_input_tokens": tool_input,
                "input_tokens": text_input + tool_input,
                "output_tokens": output,
            }

        generic = re.search(
            r"maximum context length is\s*([\d,]+)\s*tokens.*?requested about\s*([\d,]+)\s*tokens",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if generic:
            return {
                "context_limit": int(generic.group(1).replace(",", "")),
                "requested_tokens": int(generic.group(2).replace(",", "")),
            }
        return None

    def _estimate_payload_input_tokens(self, payload):
        data = payload or {}
        msg_tokens = self._estimate_messages_tokens(data.get("messages") or [])
        tools_tokens = 0
        tools = data.get("tools") or []
        model_for_encoding = str(data.get("model") or self.model or "").strip()
        if tools:
            try:
                enc = tiktoken.encoding_for_model(model_for_encoding or self.model)
            except Exception:
                enc = encoding
            try:
                tools_tokens = len(enc.encode(json.dumps(tools, default=str)))
            except Exception:
                tools_tokens = int(len(str(tools)) / 4)
        return int(msg_tokens + tools_tokens + 32)

    def _estimate_payload_input_parts(self, payload):
        data = payload or {}
        msg_tokens = self._estimate_messages_tokens(data.get("messages") or [])
        tools_tokens = 0
        tools = data.get("tools") or []
        model_for_encoding = str(data.get("model") or self.model or "").strip()
        if tools:
            try:
                enc = tiktoken.encoding_for_model(model_for_encoding or self.model)
            except Exception:
                enc = encoding
            try:
                tools_tokens = len(enc.encode(json.dumps(tools, default=str)))
            except Exception:
                tools_tokens = int(len(str(tools)) / 4)
        return {
            "message_tokens": int(msg_tokens),
            "tool_schema_tokens": int(tools_tokens),
            "input_tokens": int(msg_tokens + tools_tokens + 32),
        }

    def _estimate_input_cost_usd(self, model, input_tokens):
        model_name = str(model or self.model or "").lower()
        if "minimax" in model_name:
            # MiniMax M2.7 standard input pricing as of 2026-05: $0.30 / 1M input tokens.
            return round((float(input_tokens or 0) / 1_000_000.0) * 0.30, 6)
        if "gpt-5" in model_name:
            return round((float(input_tokens or 0) / 1_000_000.0) * 1.25, 6)
        return None

    def _emit_chat_payload_estimate(self, payload, *, phase):
        if not self.event_listener or not isinstance(payload, dict):
            return
        try:
            self._chat_payload_estimate_seq += 1
            parts = self._estimate_payload_input_parts(payload)
            model = str(payload.get("model") or self.model or "")
            event = {
                "type": "chat_payload_estimate",
                "phase": str(phase or "chat"),
                "sequence": self._chat_payload_estimate_seq,
                "model": model,
                **parts,
                "estimated_input_cost_usd": self._estimate_input_cost_usd(model, parts.get("input_tokens")),
                "message_count": len(payload.get("messages") or []),
                "tool_count": len(payload.get("tools") or []),
            }
            self.event_listener(event)
        except Exception:
            pass

    def _maybe_compact_payload_messages(self, payload, *, reason="proactive"):
        if not isinstance(payload, dict) or not self.enable_context_compaction:
            return None
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return None
        tool_schema_text = ""
        if payload.get("tools"):
            tool_schema_text = "\n[tool_schema_for_budget]\n" + self._json_dumps_safe(payload.get("tools"))
        compacted_messages, compact_meta = self._maybe_compact_messages(
            messages,
            additional_context=tool_schema_text,
        )
        if not compact_meta:
            return None
        payload["messages"] = compacted_messages
        compact_meta["reason"] = reason
        compact_meta.update(self._estimate_payload_input_parts(payload))
        self._emit_compaction_event(compact_meta)
        return compact_meta

    def _apply_proactive_output_cap(self, payload):
        model_name = str((payload or {}).get("model") or self.model or "").lower()
        if "gpt-5" not in model_name:
            return False
        if not isinstance(payload, dict):
            return False
        requested = payload.get("max_completion_tokens")
        try:
            requested = int(requested)
        except Exception:
            return False
        if requested <= 0:
            return False

        context_limit = self._context_limit_tokens()
        if context_limit <= 0:
            return False

        floor = self._min_output_tokens_floor()
        if requested <= floor:
            return False

        est_input = self._estimate_payload_input_tokens(payload)
        allowed = context_limit - est_input - self._context_retry_margin_tokens()
        if allowed <= 0:
            return False
        allowed = int(allowed)
        if allowed >= requested:
            return False
        if allowed < floor:
            return False

        payload["max_completion_tokens"] = allowed
        return True

    def _force_compact_payload_messages(self, payload):
        if not isinstance(payload, dict):
            return False
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return False

        # Keep the first message if it's system prompt/instructions.
        prefix = []
        body = list(messages)
        if body and isinstance(body[0], dict) and str(body[0].get("role") or "").lower() == "system":
            prefix = [body[0]]
            body = body[1:]

        keep_recent = max(6, min(int(self.context_compact_keep_recent or 18), 12))
        if len(body) <= keep_recent:
            return False

        removed = body[:-keep_recent]
        preserved = body[-keep_recent:]
        state_summary, open_loops, decision_log, sticky_rules = self._extract_compaction_lists(removed)
        compact_msg = {
            "role": "system",
            "content": self._render_compaction_message(
                state_summary=state_summary,
                open_loops=open_loops,
                decision_log=decision_log,
                sticky_rules=sticky_rules,
                removed_count=len(removed),
            ),
            "timestamp": datetime.now().isoformat(),
        }
        payload["messages"] = prefix + [compact_msg] + preserved
        return True

    def _attempt_context_error_recovery(self, payload, err_text):
        if not isinstance(payload, dict):
            return False

        floor = self._min_output_tokens_floor()
        current_max = payload.get("max_completion_tokens")
        try:
            current_max = int(current_max)
        except Exception:
            current_max = None

        meta = self._extract_context_error_meta(err_text)
        margin = self._context_retry_margin_tokens()

        if current_max is not None and current_max > floor:
            proposed = None
            if meta and meta.get("context_limit") and meta.get("input_tokens") is not None:
                allowed = int(meta["context_limit"]) - int(meta["input_tokens"]) - int(margin)
                if allowed > 0:
                    proposed = allowed
            if proposed is None:
                step = max(1000, int(current_max * 0.08))
                proposed = current_max - step
            proposed = max(floor, int(proposed))
            if proposed < current_max:
                payload["max_completion_tokens"] = proposed
                return True

        if self.enable_context_compaction:
            if self._force_compact_payload_messages(payload):
                return True

        # OpenRouter supports prompt transforms; apply as a final retry lever.
        if self._is_openrouter_base_url():
            extra_body = dict(payload.get("extra_body") or {})
            transforms = extra_body.get("transforms")
            if not transforms:
                extra_body["transforms"] = ["middle-out"]
                payload["extra_body"] = extra_body
                return True

        return False

    def _chat_completion_create_with_context_recovery(self, payload):
        attempts = self._context_error_max_retries()
        if not isinstance(payload, dict):
            payload = {}
        self._maybe_compact_payload_messages(payload, reason="before_chat_completion")
        self._apply_proactive_output_cap(payload)

        last_error = None
        for _ in range(attempts):
            try:
                chat_client = self._ensure_image_model_compatibility(payload)
                self._maybe_compact_payload_messages(payload, reason="retry_or_followup")
                self._apply_proactive_output_cap(payload)
                self._emit_chat_payload_estimate(payload, phase="chat_completion")
                return chat_client.chat.completions.create(**self._prepare_chat_payload(payload))
            except Exception as e:
                last_error = e
                err_text = str(e or "")
                if not self._is_context_length_error(err_text):
                    raise
                if not self._attempt_context_error_recovery(payload, err_text):
                    raise

        if last_error is not None:
            raise last_error
        raise RuntimeError("chat completion failed without a concrete exception")
    
    def _initialize_mcp_clients(self):
        """Initialize MCP client connections"""
        import asyncio
        import threading
        
        def run_mcp_setup():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._setup_mcp_connections())
            except Exception as e:
                print(f"Error initializing MCP clients: {e}")
        
        # Run MCP setup in a separate thread to avoid blocking
        mcp_thread = threading.Thread(target=run_mcp_setup, daemon=True)
        mcp_thread.start()
        mcp_thread.join(timeout=10)  # Wait up to 10 seconds for MCP setup
    
    async def _setup_mcp_connections(self):
        """Set up connections to MCP servers"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            import os
            
            for server_name, server_config in self.mcp_servers.items():
                try:
                    # Create server parameters
                    server_params = StdioServerParameters(
                        command=server_config.get('command', 'npx'),
                        args=server_config.get('args', []),
                        env=server_config.get('env', {})
                    )
                    
                    # Store connection info for later use
                    self.mcp_sessions[server_name] = {
                        'params': server_params,
                        'tools': [],
                        'connected': False
                    }
                    
                    # Try to connect and get available tools
                    await self._connect_and_get_tools(server_name, server_params)
                    
                except Exception as e:
                    print(f"Failed to initialize MCP server {server_name}: {e}")
                    
        except ImportError:
            print("MCP library not available. Install with: pip install mcp")
    
    async def _connect_and_get_tools(self, server_name, server_params):
        """Connect to an MCP server and get its tools"""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Get available tools
                    tools_response = await session.list_tools()
                    
                    for tool in tools_response.tools:
                        # Convert MCP tool to OpenAI function format
                        function_def = {
                            "type": "function",
                            "function": {
                                "name": f"mcp_{server_name}_{tool.name}",
                                "description": tool.description or f"Tool {tool.name} from {server_name}",
                                "parameters": tool.inputSchema or {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            }
                        }
                        
                        self.mcp_tools.append(function_def)
                        self.other_tools.append(function_def)
                        
                        # Create function wrapper
                        self.mcp_functions[f"mcp_{server_name}_{tool.name}"] = self._create_mcp_tool_wrapper(server_name, tool.name)
                        self.other_functions[f"mcp_{server_name}_{tool.name}"] = self.mcp_functions[f"mcp_{server_name}_{tool.name}"]
                    
                    self.mcp_sessions[server_name]['tools'] = [tool.name for tool in tools_response.tools]
                    self.mcp_sessions[server_name]['connected'] = True
                    print(f"Connected to MCP server {server_name} with {len(tools_response.tools)} tools")
                    
        except Exception as e:
            print(f"Error connecting to MCP server {server_name}: {e}")
    
    def _create_mcp_tool_wrapper(self, server_name, tool_name):
        """Create a wrapper function for an MCP tool"""
        def mcp_tool_wrapper(arguments):
            import asyncio
            import json
            
            try:
                # Parse arguments if they're a string
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                # Run the MCP tool call in a new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self._call_mcp_tool(server_name, tool_name, arguments))
                    return result
                finally:
                    loop.close()
                    
            except Exception as e:
                return f"Error calling MCP tool {server_name}.{tool_name}: {str(e)}"
        
        return mcp_tool_wrapper
    
    async def _call_mcp_tool(self, server_name, tool_name, arguments):
        """Call an MCP tool"""
        try:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client
            from mcp import types
            
            server_params = self.mcp_sessions[server_name]['params']
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    # Parse the result
                    if result.content:
                        content_parts = []
                        for content in result.content:
                            if isinstance(content, types.TextContent):
                                content_parts.append(content.text)
                            elif isinstance(content, types.ImageContent):
                                content_parts.append(f"[Image: {content.mimeType}, {len(content.data)} bytes]")
                            elif isinstance(content, types.EmbeddedResource):
                                if hasattr(content.resource, 'text'):
                                    content_parts.append(content.resource.text)
                                else:
                                    content_parts.append(f"[Resource: {content.resource.uri}]")
                        
                        return "\n".join(content_parts) if content_parts else "Tool executed successfully"
                    
                    # Check for structured content
                    if hasattr(result, 'structuredContent') and result.structuredContent:
                        return self._json_dumps_safe(result.structuredContent, indent=2)
                    
                    return "Tool executed successfully"
                    
        except Exception as e:
            return f"Error executing MCP tool: {str(e)}"
    
    def create_assistant_and_thread(self, files=None, code_interpreter=False, retrieval=False, bot_intro=None):
        tools = []
        model_descriptions = []
        valid_descriptions = []
        for config in self.configs:
            modified_tools = self.modify_tools_for_config(config)
            for tool in modified_tools:
                tools.append(tool)
            if config.model_description and config.model_description.lower() != "none":
                valid_descriptions.append(config.model_description)
        if self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Searches the web for a given query. Uses Serper when configured.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        if self.view_pages and self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "scrape_text",
                    "description": "Scrapes text from a given URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to scrape"
                            }
                        },
                        "required": ["url"]
                    }
                }
            })
        if self.other_tools is not None:
            for tool in self.other_tools:
                tools.append(tool)
        tools = self._prune_tools_for_provider(tools, [])

        if valid_descriptions:
            desc_string = " Tool information below\n---------------\n" + "\n---------------\n".join(valid_descriptions)
        else:
            desc_string = ""
        
        tool_resources = {"file_search": {"vector_store_ids": []}, "code_interpreter": {"file_ids": []}}
        if files is not None:
            for file in files:
                file_obj = self.openai_client.create(file=open(file, 'rb'), purpose='assistants')
                tool_resources["code_interpreter"]["file_ids"].append(file_obj.id)
        
        if self.assistant_id is not None:
            assistant = self.openai_client.beta.assistants.retrieve(self.assistant_id)
            if self.thread_id is not None:
                thread = self.openai_client.beta.threads.retrieve(self.thread_id)
                runs = self.openai_client.beta.threads.runs.list(self.thread_id)
                if len(runs.data) > 0:
                    latest_run = runs.data[0]
                    if latest_run.status in ["in_progress", "queued", "requires_action"]:
                        run = self.openai_client.beta.threads.runs.cancel(thread_id=self.thread_id, run_id=latest_run.id)
                        print('Cancelled run')
            else:
                thread = None
                if bot_intro is not None:
                    thread = self.openai_client.beta.threads.create(messages=[{"role": "user", "content": "Before the thread, you said " + bot_intro}])
                else:
                    thread = self.openai_client.beta.threads.create()
        else:
            if code_interpreter:
                tools.append({"type": "code_interpreter"})
            if retrieval:
                tools.append({"type": "file_search"})
            assistant = self.openai_client.beta.assistants.create(
                name=self.name,
                instructions=self.instructions + desc_string,
                model=self.model,
                tools=tools,
                tool_resources=tool_resources
            )
            self.assistant_id = assistant.id
            if bot_intro is not None:
                thread = self.openai_client.beta.threads.create(messages=[{"role": "user", "content": "Before the thread, you said " + bot_intro}])
            else:
                thread = self.openai_client.beta.threads.create()
            self.thread_id = thread.id

        return assistant, thread

    def modify_tools_for_config(self, config):
        if self.multiple_configs:
            modified_tools = []
            for tool in config.generate_tools_representation():
                if self.multiple_configs:
                    tool['function']['name'] = config.name + '-' + tool['function']['name']
                modified_tools.append(tool)
            return modified_tools
        else:
            return config.generate_tools_representation()

    def handle_old_mode(self, user_message, image_paths=None, user_tokens=None, message_id=None, files=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        print('not streaming')
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print(thread)
        
        content = user_message
        if image_paths is not None and image_paths != []:
            content = [{"type": "text", "text": user_message}]
            for image_path in image_paths:
                display_url, source_url = self._normalize_image_input(image_path)
                if not display_url:
                    continue
                image_part = {
                    "type": "image_url",
                    "image_url": {
                        "url": display_url
                    }
                }
                if source_url and source_url != display_url:
                    image_part["source_url"] = source_url
                content.append(image_part)
        file_rows = []
        for raw_file in files or []:
            if isinstance(raw_file, dict):
                file_url = str(
                    raw_file.get("url")
                    or raw_file.get("file_url")
                    or raw_file.get("href")
                    or ""
                ).strip()
                file_name = str(
                    raw_file.get("name")
                    or raw_file.get("filename")
                    or raw_file.get("title")
                    or ""
                ).strip()
            else:
                file_url = str(raw_file or "").strip()
                file_name = ""
            if not file_url:
                continue
            entry = {"url": file_url}
            if file_name:
                entry["name"] = file_name
            file_rows.append(entry)
        msg = {"role": "user", "content": content, "timestamp": datetime.now().isoformat()}
        if file_rows:
            msg["files"] = file_rows
        if message_id is not None:
            msg["message_id"] = message_id
        thread["messages"].append(msg)
        
        #print(context)
        #print(self.thread_id)
        additional_context = ""
        if self.query_memory is not None:
            if self.embedding_client is not None:
                additional_context = self.query_memory(self.thread_id, user_message, self.embedding_client, model=self.embedding_model)
            else:
                additional_context = self.query_memory(self.thread_id, user_message, self.openai_client,model=self.embedding_model)
        if additional_context is not None:
            additional_context = "\nInformation from the past that may be relevant: " + additional_context
        if self.has_file:
            kb_client = self.embedding_client or self.openai_client
            additional_context += "Information from knowledge base: " + self.read_file(self.file_identifier, user_message, kb_client)
        compacted_messages, compact_meta = self._maybe_compact_messages(thread["messages"], additional_context=additional_context)
        if compact_meta:
            thread["messages"] = compacted_messages
            self._emit_compaction_event(compact_meta)
            try:
                self.put_thread(self.thread_id, thread["messages"])
            except Exception:
                pass
        context = copy.deepcopy(thread["messages"][-self.max_messages:])
        tools = []
        if self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Searches the web for a given query. Uses Serper when configured.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        if self.view_pages and self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "scrape_text",
                    "description": "Scrapes text from a given URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to scrape"
                            }
                        },
                        "required": ["url"]
                    }
                }
            })
        if self.other_tools is not None:
            for tool in self.other_tools:
                tools.append(tool)
        model_descriptions = []
        valid_descriptions = []
        data_ = {}
        if self.raw_mode is False:
            for config in self.configs:
                modified_tools = self.modify_tools_for_config(config)
                for tool in modified_tools:
                    tools.append(tool)
                if config.model_description and config.model_description.lower() != "none":
                    valid_descriptions.append(config.model_description)
            desc_string = ""
            if len(tools) > 0:
                base_data = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions + additional_context + desc_string}] + context,
                    "tools": tools,
                    "tool_choice": "auto"
                }
                data_ = self._build_chat_data(base_data)
            else:
                base_data = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions + additional_context}] + context
                }
                data_ = self._build_chat_data(base_data)
        else:
            base_data = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.instructions + additional_context}] + context
            }
            data_ = self._build_chat_data(base_data)
        try:
            completion = self._chat_completion_create_with_context_recovery(data_)
            print(self.configs)
            if self.raw_mode == False:
                successful_tool_calls = {}
                tool_rounds = 0
                while completion.choices[0].message.role == "assistant":
                    current_message = completion.choices[0].message
                    raw_message_text = self._content_to_text(getattr(current_message, "content", ""))
                    legacy_tool_calls = []
                    native_tool_calls = list(getattr(current_message, "tool_calls", None) or [])
                    if not native_tool_calls:
                        legacy_tool_calls = self._parse_legacy_tool_calls(raw_message_text)
                    tool_calls_to_run = native_tool_calls or legacy_tool_calls
                    if not tool_calls_to_run:
                        break
                    tool_outputs = []
                    async_tool_names = []
                    for tool_call in tool_calls_to_run:
                        if hasattr(tool_call, "function"):
                            tool_name = tool_call.function.name
                            tool_args = tool_call.function.arguments
                            tool_id = tool_call.id
                        else:
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("arguments")
                            tool_id = tool_call.get("id")
                        try:
                            signature = self._tool_call_signature(tool_name, tool_args)
                            if signature in successful_tool_calls:
                                result = self._duplicate_tool_output(
                                    tool_name,
                                    tool_args,
                                    successful_tool_calls[signature],
                                )
                            else:
                                result = self.execute_function(tool_name, tool_args, user_tokens)
                                if self._tool_result_success(result):
                                    successful_tool_calls[signature] = result
                            output = {
                                "tool_call_id": tool_id,
                                "output": self._json_dumps_safe(result),
                                "tool_name": tool_name,
                                "tool_arguments": tool_args
                            }
                        except Exception as e:
                            try:
                                logger.exception("Error executing tool %s", tool_name)
                            except Exception:
                                pass
                            result = {"success": False, "error": str(e)}
                            output = {
                                "tool_call_id": tool_id,
                                "output": self._json_dumps_safe(result),
                                "tool_name": tool_name,
                                "tool_arguments": tool_args,
                            }
                        tool_outputs.append(output)
                        if self.event_listener is not None:
                            self.event_listener(output)
                        if tool_name in self.async_tools:
                            async_tool_names.append(tool_name)
                    compact_outputs = self._compact_tool_outputs_for_context(tool_outputs)
                    tool_image_urls = self._extract_tool_generated_image_urls(tool_outputs)
                    tool_rounds += 1
                    force_no_more_tools = tool_rounds >= self._max_tool_rounds()
                    data_['messages'] = data_['messages'] + [{"role": "system", "content": self._tool_output_followup_hint(compact_outputs, force_no_more_tools=force_no_more_tools)}]
                    thread["messages"].append({
                        "role": "system",
                        "content": self._tool_output_followup_hint(compact_outputs, force_no_more_tools=force_no_more_tools),
                    })
                    image_msg = self._tool_image_context_message(tool_image_urls)
                    if image_msg:
                        data_["messages"].append(image_msg)
                    async_hint = self._async_tool_system_hint(async_tool_names)
                    if async_hint:
                        data_['messages'].append({"role": "system", "content": async_hint})
                    if force_no_more_tools:
                        data_["tools"] = None
                        data_.pop("tool_choice", None)

                    completion = self._chat_completion_create_with_context_recovery(data_)
            print(completion.choices[0].message)
            raw_response_message = self._content_to_text(completion.choices[0].message.content)
            response_message, think_reasoning = self._strip_think_blocks(raw_response_message)
            response_message, _legacy_tool_blocks = self._strip_legacy_tool_call_blocks(response_message)
            direct_reasoning = self._extract_reasoning_from_obj(completion.choices[0].message)
            combined_reasoning = self._merge_reasoning_text(direct_reasoning, think_reasoning)
            print(response_message)
            assistant_row = {"role": "assistant", "content": response_message}
            if combined_reasoning:
                assistant_row["reasoning"] = combined_reasoning
                self._emit_reasoning_event(combined_reasoning)
            thread["messages"].append(assistant_row)
            self.put_thread(self.thread_id, thread["messages"])
            if self.save_memory is not None:
                if self.embedding_model is None:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": response_message}), self.embedding_client)).start()
                    else:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": response_message}), self.openai_client)).start()
                else:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": response_message}), self.embedding_client), 
                        kwargs={'model': self.embedding_model}).start()
                    else:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": response_message}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
            return response_message
        except Exception as e:
            print(e)
            return "Error "+str(e)

    def handle_old_mode_streaming(self, user_message, image_paths=None, user_tokens=None, files=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        #print(thread)
        
        content = [{"type": "text", "text": user_message}]
        if image_paths is not None:
            for image_path in image_paths:
                display_url, source_url = self._normalize_image_input(image_path)
                if not display_url:
                    continue
                image_part = {
                    "type": "image_url",
                    "image_url": {
                        "url": display_url
                    }
                }
                if source_url and source_url != display_url:
                    image_part["source_url"] = source_url
                content.append(image_part)
        file_rows = []
        for raw_file in files or []:
            if isinstance(raw_file, dict):
                file_url = str(
                    raw_file.get("url")
                    or raw_file.get("file_url")
                    or raw_file.get("href")
                    or ""
                ).strip()
                file_name = str(
                    raw_file.get("name")
                    or raw_file.get("filename")
                    or raw_file.get("title")
                    or ""
                ).strip()
            else:
                file_url = str(raw_file or "").strip()
                file_name = ""
            if not file_url:
                continue
            entry = {"url": file_url}
            if file_name:
                entry["name"] = file_name
            file_rows.append(entry)
        timestamp = datetime.now().isoformat()
        user_row = {"role": "user", "content": content, "timestamp": timestamp}
        if file_rows:
            user_row["files"] = file_rows
        thread["messages"].append(user_row)
        try:
            self.put_thread(self.thread_id, thread["messages"])
        except Exception:
            pass
        additional_context = ""
        if self.query_memory is not None:
            if self.embedding_client is not None:
                additional_context = self.query_memory(self.thread_id, user_message, self.embedding_client,model=self.embedding_model)
            else:
                additional_context = self.query_memory(self.thread_id, user_message, self.openai_client,model=self.embedding_model)
        if additional_context is not None:
            additional_context = "\nInformation from the past that may be relevant: " + additional_context
        if self.has_file:
            kb_client = self.embedding_client or self.openai_client
            additional_context += "Information from knowledge base: " + self.read_file(self.file_identifier, user_message, kb_client)
        compacted_messages, compact_meta = self._maybe_compact_messages(thread["messages"], additional_context=additional_context)
        if compact_meta:
            thread["messages"] = compacted_messages
            self._emit_compaction_event(compact_meta)
        use_legacy_trim = not (self.enable_context_compaction and int(self.context_budget_tokens or 0) > 0)
        if use_legacy_trim and len(thread["messages"]) > self.max_messages:
            thread["messages"] = thread["messages"][-self.max_messages:]
        try:
            self.put_thread(self.thread_id, thread["messages"])
        except Exception:
            pass
        
        tools = []
        if self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Searches the web for a given query. Uses Serper when configured.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        if self.view_pages and self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "scrape_text",
                    "description": "Scrapes text from a given URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to scrape"
                            }
                        },
                        "required": ["url"]
                    }
                }
            })
        if self.other_tools is not None:
            for tool in self.other_tools:
                tools.append(tool)
        if len(tools) == 0:
            tools = None
        base_data = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.instructions + additional_context}] + thread["messages"],
            "stream": True,
            "tools": tools,
            "tool_choice": "auto" if tools is not None and len(tools) > 0 else None
        }
        data_ = self._build_chat_data(base_data)

        #print(data_)
        done = False
        successful_tool_calls = {}
        tool_rounds = 0
        try:
            while not done:
                if self.stop_check and self.stop_check():
                    return
                completion = self._chat_completion_create_with_context_recovery(data_)

                result = ""
                raw_result = ""
                think_stream_state = {"pending": "", "in_think": False}
                legacy_tool_stream_state = {"pending": "", "in_tool_call": False}
                reasoning_chunks = []
                tool_calls = {}
                tool_call_ids_by_index = {}
                finish_reason = None

                for response_chunk in completion:
                    if self.stop_check and self.stop_check():
                        return
                    try:
                        choices = getattr(response_chunk, "choices", None) or []
                        if not choices:
                            continue
                        first_choice = choices[0]
                        finish_reason = getattr(first_choice, "finish_reason", None) or finish_reason
                        delta = getattr(first_choice, "delta", None)
                        if delta is None:
                            continue

                        delta_reasoning = self._extract_reasoning_from_obj(delta)
                        if delta_reasoning:
                            reasoning_chunks.append(delta_reasoning)

                        delta_content = getattr(delta, "content", None)
                        if delta_content is not None:
                            if isinstance(delta_content, list):
                                text_parts = []
                                for part in delta_content:
                                    if isinstance(part, dict):
                                        text_parts.append(str(part.get("text") or ""))
                                    else:
                                        text_parts.append(str(getattr(part, "text", "") or ""))
                                delta_text = "".join(text_parts)
                            else:
                                delta_text = str(delta_content)
                            if delta_text:
                                raw_result += delta_text
                                visible_chunk, reasoning_chunk = self._consume_stream_think_chunk(delta_text, think_stream_state)
                                if reasoning_chunk:
                                    reasoning_chunks.append(reasoning_chunk)
                                if visible_chunk:
                                    visible_chunk = self._consume_stream_legacy_tool_call_chunk(
                                        visible_chunk,
                                        legacy_tool_stream_state,
                                    )
                                    if visible_chunk:
                                        result += visible_chunk
                                        yield visible_chunk

                        delta_tool_calls = getattr(delta, "tool_calls", None) or []
                        for tool_call in delta_tool_calls:
                            call_id = tool_call.id
                            try:
                                idx = tool_call.index
                            except Exception:
                                idx = None
                            if call_id:
                                if idx is not None:
                                    tool_call_ids_by_index[idx] = call_id
                            else:
                                if idx is not None and idx in tool_call_ids_by_index:
                                    call_id = tool_call_ids_by_index[idx]
                                elif len(tool_calls) == 1:
                                    call_id = next(iter(tool_calls.keys()))
                                else:
                                    call_id = f"call_{len(tool_calls)}"
                            if call_id not in tool_calls:
                                tool_calls[call_id] = {"name": None, "arguments": ""}
                            if tool_call.function.name:
                                tool_calls[call_id]["name"] = tool_call.function.name
                            if tool_call.function.arguments:
                                tool_calls[call_id]["arguments"] += tool_call.function.arguments
                    except Exception:
                        try:
                            logger.exception("Skipping malformed streaming chunk")
                        except Exception:
                            pass
                        continue

                legacy_tool_calls = []
                if not tool_calls:
                    legacy_tool_calls = self._parse_legacy_tool_calls(raw_result)
                    for legacy_call in legacy_tool_calls:
                        call_id = str(legacy_call.get("id") or f"legacy_call_{len(tool_calls)}")
                        tool_calls[call_id] = {
                            "name": legacy_call.get("name"),
                            "arguments": legacy_call.get("arguments") or "{}",
                        }

                if tool_calls:
                    tool_outputs = []
                    assistant_tool_calls = []
                    async_tool_names = []
                    for call_id, info in tool_calls.items():
                        if self.stop_check and self.stop_check():
                            return
                        tool_name = info.get("name")
                        tool_args = info.get("arguments") or "{}"
                        if not tool_name:
                            continue
                        tool_name_for_mess = tool_name.replace('_', ' ')
                        if tool_name == 'view_page':
                            tool_name_for_mess = 'view a page'
                        if tool_name == 'transfer':
                            tool_name_for_mess = 'transfer your call'
                        if self.emit_tool_preamble:
                            yield "Hang on, gotta " + tool_name_for_mess
                        try:
                            signature = self._tool_call_signature(tool_name, tool_args)
                            if signature in successful_tool_calls:
                                output_result = self._duplicate_tool_output(
                                    tool_name,
                                    tool_args,
                                    successful_tool_calls[signature],
                                )
                            else:
                                output_result = self.execute_function(tool_name, tool_args, user_tokens)
                                if self._tool_result_success(output_result):
                                    successful_tool_calls[signature] = output_result
                            output = {
                                "tool_call_id": call_id,
                                "output": self._json_dumps_safe(output_result),
                                "tool_name": tool_name,
                                "tool_arguments": tool_args
                            }
                        except Exception as e:
                            try:
                                logger.exception("Error executing tool %s", tool_name)
                            except Exception:
                                pass
                            output = {
                                "tool_call_id": call_id,
                                "output": self._json_dumps_safe({"error": str(e)}),
                                "tool_name": tool_name,
                                "tool_arguments": tool_args
                            }
                            output_result = {"error": str(e)}
                        tool_outputs.append(output)
                        assistant_tool_calls.append({
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": tool_args}
                        })
                        if self.event_listener is not None:
                            self.event_listener(output)
                        if tool_name in self.async_tools:
                            async_tool_names.append(tool_name)
                    compact_outputs = self._compact_tool_outputs_for_context(tool_outputs)
                    tool_image_urls = self._extract_tool_generated_image_urls(tool_outputs)

                    data_['messages'] = data_['messages'] + [
                        {"role": "assistant", "content": None, "tool_calls": assistant_tool_calls}
                    ]
                    for output in compact_outputs:
                        data_['messages'].append({
                            "role": "tool",
                            "tool_call_id": output["tool_call_id"],
                            "content": output["output"]
                        })
                    tool_rounds += 1
                    force_no_more_tools = tool_rounds >= self._max_tool_rounds()
                    data_['messages'].append({
                        "role": "system",
                        "content": self._tool_output_followup_hint(
                            compact_outputs,
                            force_no_more_tools=force_no_more_tools,
                            include_outputs=False,
                        ),
                    })
                    try:
                        thread["messages"].append({
                            "role": "system",
                            "content": self._tool_output_followup_hint(compact_outputs, force_no_more_tools=force_no_more_tools),
                        })
                        self.put_thread(self.thread_id, thread["messages"])
                    except Exception:
                        pass
                    image_msg = self._tool_image_context_message(tool_image_urls)
                    if image_msg:
                        data_["messages"].append(image_msg)
                    async_hint = self._async_tool_system_hint(async_tool_names)
                    if async_hint:
                        data_['messages'].append({"role": "system", "content": async_hint})
                    if force_no_more_tools:
                        data_["tools"] = None
                        data_.pop("tool_choice", None)
                    done = False
                    continue

                done = True

                tail_visible, tail_reasoning = self._flush_stream_think_state(think_stream_state)
                if tail_visible:
                    tail_visible = self._consume_stream_legacy_tool_call_chunk(
                        tail_visible,
                        legacy_tool_stream_state,
                    )
                legacy_tail_visible = self._flush_stream_legacy_tool_call_state(legacy_tool_stream_state)
                if legacy_tail_visible:
                    tail_visible = (tail_visible or "") + legacy_tail_visible
                if tail_reasoning:
                    reasoning_chunks.append(tail_reasoning)
                if tail_visible:
                    result += tail_visible
                    yield tail_visible

            cleaned_result, think_reasoning = self._strip_think_blocks(raw_result or result)
            cleaned_result, _legacy_tool_blocks = self._strip_legacy_tool_call_blocks(cleaned_result)
            if cleaned_result:
                result = cleaned_result
            combined_reasoning = self._merge_reasoning_text("\n\n".join(reasoning_chunks), think_reasoning)

            assistant_row = {"role": "assistant", "content": result}
            if combined_reasoning:
                assistant_row["reasoning"] = combined_reasoning
                self._emit_reasoning_event(combined_reasoning)

            thread["messages"].append(assistant_row)
            #self.put_thread(self.thread_id, thread["messages"])
            threading.Thread(target=self.put_thread, args=(self.thread_id, thread["messages"])).start()
            if self.save_memory is not None:
                if self.embedding_model is None:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": result}), self.embedding_client)).start()
                    else:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": result}), self.openai_client)).start()
                else:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": result}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
                    else:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, self._json_dumps_safe({"input": user_message, "output": result}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
            return result
        except Exception as e:
            # NOTE: This is a streaming generator; returning a string here results in an empty
            # stream for callers (StopIteration.value is ignored). Yield a short, safe error
            # so the UI never looks "stuck", while logging the full traceback server-side.
            try:
                logger.exception("Assistant streaming failure (old_mode)")
            except Exception:
                pass
            err = f"Internal error while generating response: {type(e).__name__}: {str(e)[:500]}"
            yield err
            return
    def delete_message_assistant(self, message_id):
        self.openai_client.beta.threads.messages.delete(thread_id=self.thread.id, message_id=message_id)
        return "Message deleted"

    def get_assistant_response(self, message, files=None, image_paths=None, user_tokens=None, message_id=None, store_mid=None):
        if self.old_mode:
            if self.streaming:
                return self.handle_old_mode_streaming(message, image_paths=image_paths, user_tokens=user_tokens, files=files)
            return self.handle_old_mode(message, image_paths=image_paths, user_tokens=user_tokens, message_id=message_id, files=files)
        
        attachments = []
        if files is not None:
            for file_path in files:
                file_obj = self.openai_client.create(file=open(file_path, 'rb'), purpose='assistants')
                attachments.append({"file_id": file_obj.id, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]})
        
        content = [{"type": "text", "text": message}]
        if image_paths is not None:
            for image_path in image_paths:
                display_url, source_url = self._normalize_image_input(image_path)
                if not display_url:
                    continue
                image_part = {
                    "type": "image_url",
                    "image_url": {
                        "url": display_url
                    }
                }
                if source_url and source_url != display_url:
                    image_part["source_url"] = source_url
                content.append(image_part)
        
        message_obj = self.openai_client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content,
            attachments=attachments if attachments else None
        )
        if store_mid is not None:
            store_mid(message_obj.id, message_id, self.thread.id)

        run = self.openai_client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
        )
        
        print("Waiting for response")
        print(run.id)
        completed = False
        successful_tool_calls = {}
        while not completed:
            run_ = self.openai_client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
            if run_.status == "completed":
                break
            elif run_.status == "failed":
                print("Run failed")
                break
            elif run_.status == "cancelled":
                print("Run cancelled")
                break
            elif run_.status == "requires_action":
                print("Run requires action")
                tool_calls = run_.required_action.submit_tool_outputs.tool_calls
                tool_outputs = []
                for tool_call in tool_calls:
                    if self.event_listener is not None:
                        tool_call_dict = tool_call.__dict__.copy()
                        tool_call_dict['function'] = str(tool_call_dict['function'])
                        print(tool_call_dict)
                        self.event_listener(tool_call_dict)
                    if tool_call.type == "function":
                        user_token = None
                        if user_tokens is not None:
                            if self.multiple_configs:
                                user_token = user_tokens.get(tool_call.function.name.split('-', 1)[0])
                            else:
                                user_token = user_tokens[self.configs[0].name]
                        signature = self._tool_call_signature(tool_call.function.name, tool_call.function.arguments)
                        if signature in successful_tool_calls:
                            result = self._duplicate_tool_output(
                                tool_call.function.name,
                                tool_call.function.arguments,
                                successful_tool_calls[signature],
                            )
                        else:
                            result = self.execute_function(tool_call.function.name, tool_call.function.arguments, user_token=user_token)
                            if self._tool_result_success(result):
                                successful_tool_calls[signature] = result
                        output = {
                            "tool_call_id": tool_call.id,
                            "output": self._json_dumps_safe(result)
                        }
                        if self.event_listener is not None:
                            self.event_listener(output)
                        tool_outputs.append(output)
                run__ = self.openai_client.beta.threads.runs.submit_tool_outputs(thread_id=self.thread.id, run_id=run.id, tool_outputs=tool_outputs)
            time.sleep(1)
        run_ = self.openai_client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=run.id)
        messages = self.openai_client.beta.threads.messages.list(thread_id=self.thread.id)
        print(messages.data[0].content[0].text.value)
        return messages.data[0].content[0].text.value

    def get_entire_conversation(self):
        messages = self.openai_client.beta.threads.messages.list(thread_id=self.thread.id)
        return messages.data

    def execute_function(self, function_name, arguments, user_token=None):
        try:
            x = json.loads(arguments)
        except Exception as e:
            return "JSON not valid"
        normalized_function_name = str(function_name or "").strip()
        if not normalized_function_name:
            return {"success": False, "error": "function_name_required"}
        if normalized_function_name == "search_google":
            return search_google(x["query"])
        if normalized_function_name == "scrape_text":
            target_url = (
                x.get("url")
                or x.get("link")
                or x.get("source_url")
                or x.get("href")
                or ""
            )
            if not str(target_url or "").strip():
                return {"success": False, "error": "url_required", "message": "scrape_text requires a url"}
            return scrape_text(str(target_url).strip(), self.search_window)
        def _local_function_candidates(name):
            candidates = []

            def _add(candidate):
                normalized = str(candidate or "").strip()
                if normalized and normalized not in candidates:
                    candidates.append(normalized)

            normalized_name = str(name or "").strip()
            _add(normalized_name)
            if normalized_name.startswith("operator_"):
                _add(normalized_name[len("operator_"):])
            else:
                _add(f"operator_{normalized_name}")

            route_match = re.match(
                r"^(?P<base>.+?)_route_(?P<method>get|post|put|patch|delete|head|options)$",
                normalized_name,
                flags=re.IGNORECASE,
            )
            if route_match:
                route_base = str(route_match.group("base") or "").strip()
                _add(route_base)
                if route_base.startswith("operator_"):
                    _add(route_base[len("operator_"):])
                else:
                    _add(f"operator_{route_base}")
            return candidates

        available_other_functions = self.other_functions if isinstance(self.other_functions, dict) else {}
        other_tool_names = []
        if self.other_tools is not None:
            other_tool_names = [
                str(((tool or {}).get('function') or {}).get('name') or '').strip()
                for tool in self.other_tools
                if isinstance(tool, dict)
            ]
            if normalized_function_name in other_tool_names:
                func_to_call = available_other_functions.get(normalized_function_name)
                if not callable(func_to_call):
                    return {
                        "success": False,
                        "error": f"function_handler_not_found:{normalized_function_name}",
                    }
                return func_to_call(x)
        for candidate_name in _local_function_candidates(normalized_function_name):
            func_to_call = available_other_functions.get(candidate_name)
            if callable(func_to_call):
                return func_to_call(x)
        if self.multiple_configs and '-' in normalized_function_name:
            config_name, actual_function_name = normalized_function_name.split('-', 1)
            config = next((cfg for cfg in self.configs if cfg.name == config_name), None)
        else:
            actual_function_name = normalized_function_name
            config = self.configs[0] if self.configs else None

        if not config:
            return {
                "success": False,
                "error": "function_not_found",
                "function_name": normalized_function_name,
                "tried_local_names": _local_function_candidates(normalized_function_name),
                "configured_api_count": len(self.configs or []),
            }

        arguments = x
        is_json = config.is_json
        print(config.name)
        print(actual_function_name)
        try:
            request = config.make_api_call_by_operation_id(actual_function_name, params=arguments, is_json=is_json, user_token=user_token)
            print(request)
            print(request.status_code)
            print(request.reason)
            try:
                return request.json() + "\n " + str(request.status_code) + " " + request.reason
            except Exception as e:
                return request.text + "\n " + str(request.status_code) + " " + request.reason
        except Exception as e:
            print(e)
            try:
                split = actual_function_name.split("-")
                method = split[1]
                if method.upper() == "GET" or method.upper() == "DELETE":
                    is_json = False
                path = split[0]
                request = config.make_api_call_by_path(path, method.upper(), params=arguments, is_json=is_json, user_token=user_token)
                print(request)
                print(request.status_code)
                print(request.reason)
                print(request.text)
                try:
                    return request.json() + "\n " + str(request.status_code) + " " + request.reason
                except Exception as e:
                    return request.text + "\n " + str(request.status_code) + " " + request.reason
            except Exception as e:
                print(e)
                import traceback
                traceback.print_exc()
                try:
                    request = config.make_api_call_by_path('/' + path, method.upper(), params=arguments, is_json=is_json, user_token=user_token)
                    print(request)
                    print(request.text)
                    print(request.status_code)
                    print(request.reason)
                    try:
                        return request.json() + "\n " + str(request.status_code) + " " + request.reason
                    except Exception as e:
                        return request.text + "\n " + str(request.status_code) + " " + request.reason
                except Exception as e:
                    print(e)
                    return "Error"
        return "Error"

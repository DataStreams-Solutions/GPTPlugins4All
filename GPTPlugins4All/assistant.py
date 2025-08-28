import os
import time
import json
from dotenv import load_dotenv
import uuid
import threading
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import base64
import copy
from datetime import datetime

load_dotenv()

from typing import Optional
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


cost_dict = {'o3-mini': 110/1000000, 'gpt-3.5-turbo': 1/1000000, 'o1': 60000/1000000, 'gpt-4o': 1000/1000000, 'gpt-4o-mini': 60/1000000}
def search_google(query, num_results=6):
    search_results = []
    res_string = ""
    print("Searching google for: " + query)
    try:
        for j in search(query, num_results=num_results, advanced=True,lang="en",
            region="US"):
            search_results.append(j)
            res_string += j.url + " - " + j.title + " - "+j.description
            res_string += "\n\n"
        return "Results from google search: " + query + "\n" + res_string
    except Exception as e:
        return "Error: " + str(e)

def leftTruncate(text, length):
    encoded = encoding.encode(text)
    num = len(encoded)
    if num > length:
        return encoding.decode(encoded[num - length:])
    else:
        return text

def scrape_text(url, length):
    import re
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    response = requests.get(url, verify=False, headers=headers)
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
        print(f"Falling back to headless browser for {url} (content_small: {content_too_small}, has_links: {has_links})")
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)
                page.wait_for_selector("body", timeout=30000)
                response_text = page.content()
                browser.close()
                print(f"Successfully rendered page with headless browser, content length: {len(response_text)}")
        except Exception as e:
            print(f"Error during fallback rendering: {e}")
            # Continue with original response if fallback fails

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
    def __init__(self, configs, name, instructions, model, assistant_id=None, thread_id=None, embedding_key=None,event_listener=None, openai_key=None, files=None, code_interpreter=False, retrieval=False, is_json=None, old_mode=False, max_tokens=None, bot_intro=None, get_thread=None, put_thread=None, save_memory=None, query_memory=None, max_messages=4, raw_mode=False, streaming=False, has_file=False, file_identifier=None, read_file=None, search_enabled=False, view_pages=False, search_window=1000, other_tools=None, other_functions={}, embedding_model=None, base_url=None, suggest_responses=False, api_calls=[], sources=None, initial_suggestions=None):
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
        self.other_tools = other_tools or []
        self.other_functions = other_functions or {}
        self.initial_suggestions = initial_suggestions
        suggestions_str = ''
        if initial_suggestions is not None:
            suggestions_str = json.dumps(initial_suggestions)

        # Generate tools and functions from api_calls
        more_tools, more_functions = generate_tools_from_api_calls(api_calls)
        self.other_tools += more_tools
        self.other_functions.update(more_functions)

        self.suggest_responses = suggest_responses
        if self.suggest_responses:
            self.instructions += "\nIn addition to the above, *always* give the user potential replies (eg quick-replies) to follow up with in this format: \n[\"response1\", \"response2\", \"response3\"]"
            print('suggestions enabled')
        if is_json is not None:
            self.is_json = is_json
        if openai_key is None:
            if base_url is None or base_url == '' or base_url == 'https://api.openai.com':
                self.openai_client = OpenAI()
            else:
                self.openai_client = OpenAI(base_url=base_url)
                self.embedding_client = OpenAI(api_key=embedding_key)
        else:
            if base_url is None or base_url == '' or base_url == 'https://api.openai.com':
                self.openai_client = OpenAI(api_key=openai_key)
            else:
                self.openai_client = OpenAI(api_key=openai_key, base_url=base_url)
                self.embedding_client = OpenAI(api_key=embedding_key)
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

    def add_file(self, file):
        file = self.openai_client.create(
            file=open(file, 'rb'),
            purpose='assistants'
        )
        self.openai_client.beta.assistants.update(self.assistant_id, tool_resources={"code_interpreter": {"file_ids": [file.id]}})
    
    def _build_chat_data(self, base_data):
        """Helper method to build chat completion data with correct token parameter"""
        data = base_data.copy()
        if 'gpt-5' in self.model.lower():
            if self.max_completion_tokens is not None:
                data["max_completion_tokens"] = self.max_completion_tokens
        else:
            if self.max_tokens is not None:
                data["max_tokens"] = self.max_tokens
        return data
    
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
                    "description": "Searches Google for a given query",
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
        if self.view_pages:
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

    def handle_old_mode(self, user_message, image_paths=None, user_tokens=None, message_id=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        print('not streaming')
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print(thread)
        
        content = user_message
        if image_paths is not None and image_paths != []:
            for image_path in image_paths:
                #base64_image = encode_image(image_path)
                print(image_path)
                content = [{"type": "text", "text": user_message}]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                })
        msg = {"role": "user", "content": content, "timestamp": datetime.now().isoformat()}
        if message_id is not None:
            msg["message_id"] = message_id
        thread["messages"].append(msg)
        context = copy.deepcopy(thread["messages"][-self.max_messages:])
        
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
            additional_context += "Information from knowledge base: " + self.read_file(self.file_identifier, user_message, self.openai_client)
        tools = []
        if self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Searches Google for a given query",
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
        if self.view_pages:
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
            completion = self.openai_client.chat.completions.create(**data_)
            print(self.configs)
            if self.raw_mode == False:
                while completion.choices[0].message.role == "assistant" and completion.choices[0].message.tool_calls:
                    tool_outputs = []
                    for tool_call in completion.choices[0].message.tool_calls:
                        result = self.execute_function(tool_call.function.name, tool_call.function.arguments, user_tokens)
                        output = {
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(result),
                            "tool_name": tool_call.function.name,
                            "tool_arguments": tool_call.function.arguments
                        }
                        tool_outputs.append(output)
                        if self.event_listener is not None:
                            self.event_listener(output)
                    data_['messages'] = data_['messages'] + [{"role": "system", "content": "Tool outputs from most recent attempt" + json.dumps(tool_outputs) + "\n If the above indicates an error, change the input and try again"}]
                    thread["messages"].append({"role": "system", "content": "Tool outputs from most recent attempt: " + json.dumps(tool_outputs)})

                    completion = self.openai_client.chat.completions.create(**data_)

            response_message = completion.choices[0].message.content
            print(response_message)
            thread["messages"].append({"role": "assistant", "content": response_message})
            self.put_thread(self.thread_id, thread["messages"])
            if self.save_memory is not None:
                if self.embedding_model is None:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.embedding_client)).start()
                    else:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.openai_client)).start()
                else:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.embedding_client), 
                        kwargs={'model': self.embedding_model}).start()
                    else:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
            return response_message
        except Exception as e:
            print(e)
            return "Error "+str(e)

    def handle_old_mode_streaming(self, user_message, image_paths=None, user_tokens=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print('got here')
        #print(thread)
        print('streaming -------------------')
        
        content = [{"type": "text", "text": user_message}]
        if image_paths is not None:
            for image_path in image_paths:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                })
        timestamp = datetime.now().isoformat()
        thread["messages"].append({"role": "user", "content": content, "timestamp": timestamp})
        if len(thread["messages"]) > self.max_messages:
            thread["messages"] = thread["messages"][-self.max_messages:]
        additional_context = ""
        if self.query_memory is not None:
            if self.embedding_client is not None:
                additional_context = self.query_memory(self.thread_id, user_message, self.embedding_client,model=self.embedding_model)
            else:
                additional_context = self.query_memory(self.thread_id, user_message, self.openai_client,model=self.embedding_model)
        if additional_context is not None:
            additional_context = "\nInformation from the past that may be relevant: " + additional_context
        if self.has_file:
            additional_context += "Information from knowledge base: " + self.read_file(self.file_identifier, user_message, self.openai_client)
        
        tools = []
        if self.search_enabled:
            tools.append({
                "type": "function",
                "function": {
                    "name": "search_google",
                    "description": "Searches Google for a given query",
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
        if self.view_pages:
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
        try:
            while not done:
                completion = self.openai_client.chat.completions.create(**data_)
                #print(completion)

                result = ""
                tool_calls = {}
                arg_acc =''
                tool_name = ''
                for response_chunk in completion:
                    delta = response_chunk.choices[0].delta
                    if delta.content is not None:
                        done = True
                        result += delta.content
                        yield delta.content
                    
                    if delta.tool_calls:
                        tool_outputs = []
                        #yield "Hang on, gotta do some stuff"
                        for tool_call in delta.tool_calls:
                            tool_calls[tool_call.id] = tool_call
                            print('tool call')
                            print(tool_call)
                            
                            arg_acc += tool_call.function.arguments
                            #check that arguments are complete
                            if tool_call.function.name != None:
                                tool_name = tool_call.function.name
                                #if self.event_listener is not None:
                                #    self.event_listener("Hang on, gotta look up some stuff")
                                #    sys_message = "Hang on, gotta "+ tool_name
                                #    yield sys_message
                                #    print('telling system to hang on')
                            print(arg_acc)
                            if len(arg_acc) > 0 and arg_acc[-1] != '}':
                                continue
                            #check that it is a valid json
                            
                            try:
                                x = json.loads(arg_acc)
                            except Exception as e:
                                continue
                            #replace underscore with space
                            tool_name_for_mess = tool_name.replace('_', ' ')
                            if tool_name == 'view_page':
                                tool_name_for_mess = 'view a page'
                            if tool_name == 'transfer':
                                tool_name_for_mess = 'transfer your call'
                            sys_message = "Hang on, gotta "+ tool_name_for_mess
                            yield sys_message
                            try:
                                print(arg_acc)
                                
                                result = self.execute_function(tool_name, arg_acc, user_tokens)
                                arg_acc = ''
                                tool_name = ''
                                
                                print(result)
                                output = {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps(result),
                                    "tool_name": tool_call.function.name,
                                    "tool_arguments": tool_call.function.arguments
                                }
                                tool_outputs.append(output)
                                if self.event_listener is not None:
                                    self.event_listener(output)
                            except Exception as e:
                                print(f"Error executing tool: {e}")
                                output = {
                                    "tool_call_id": tool_call.id,
                                    "output": json.dumps({"error": str(e)}),
                                    "tool_name": tool_call.function.name,
                                    "tool_arguments": tool_call.function.arguments
                                }
                                tool_outputs.append(output)
                                if self.event_listener is not None:
                                    self.event_listener(output)
                            
                        data_['messages'] = data_['messages'] + [{"role": "system", "content": "Tool outputs from most recent attempt: " + json.dumps(tool_outputs)}]
                    #completion = self.openai_client.chat.completions.create(**data_)

            thread["messages"].append({"role": "assistant", "content": result})
            #self.put_thread(self.thread_id, thread["messages"])
            threading.Thread(target=self.put_thread, args=(self.thread_id, thread["messages"])).start()
            if self.save_memory is not None:
                if self.embedding_model is None:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": result}), self.embedding_client)).start()
                    else:
                        threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": result}), self.openai_client)).start()
                else:
                    if self.embedding_client is not None:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, json.dumps({"input": user_message, "output": result}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
                    else:
                        threading.Thread(target=self.save_memory, 
                        args=(self.thread_id, json.dumps({"input": user_message, "output": result}), self.openai_client), 
                        kwargs={'model': self.embedding_model}).start()
            return result
        except Exception as e:
            print(e)
            return "Error "+str(e)
    def delete_message_assistant(self, message_id):
        self.openai_client.beta.threads.messages.delete(thread_id=self.thread.id, message_id=message_id)
        return "Message deleted"

    def get_assistant_response(self, message, files=None, image_paths=None, user_tokens=None, message_id=None, store_mid=None):
        if self.old_mode:
            if self.streaming:
                return self.handle_old_mode_streaming(message, image_paths=image_paths, user_tokens=user_tokens)
            return self.handle_old_mode(message, image_paths=image_paths, user_tokens=user_tokens, message_id=message_id)
        
        attachments = []
        if files is not None:
            for file_path in files:
                file_obj = self.openai_client.create(file=open(file_path, 'rb'), purpose='assistants')
                attachments.append({"file_id": file_obj.id, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]})
        
        content = [{"type": "text", "text": message}]
        if image_paths is not None:
            for image_path in image_paths:
                #base64_image = encode_image(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                })
        
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
                        result = self.execute_function(tool_call.function.name, tool_call.function.arguments, user_token=user_token)
                        output = {
                            "tool_call_id": tool_call.id,
                            "output": json.dumps(result)
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
            print(e)
            return "JSON not valid"
        print(function_name)
        print(arguments)
        if function_name == "search_google":
            return search_google(x["query"])
        if function_name == "scrape_text":
            return scrape_text(x["url"], self.search_window)
        other_tool_names = []
        if self.other_tools is not None:
            other_tool_names = [tool['function']['name'] for tool in self.other_tools]
            if function_name in other_tool_names:
                func_to_call = self.other_functions[function_name]
                return func_to_call(x)
        if self.multiple_configs and '-' in function_name:
            config_name, actual_function_name = function_name.split('-', 1)
            config = next((cfg for cfg in self.configs if cfg.name == config_name), None)
        else:
            actual_function_name = function_name
            config = self.configs[0]

        if not config:
            return "Configuration not found for function: " + function_name

        arguments = json.loads(arguments)
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
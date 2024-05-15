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

def search_google(query):
    search_results = []
    res_string = ""

    for j in search(query, num_results=6, advanced=True):
        search_results.append(j)
        res_string += j.url + " - " + j.title + " - "+j.description
        res_string += "\n\n"
    return "Results from google search: " + query + "\n" + res_string

def leftTruncate(text, length):
    encoded = encoding.encode(text)
    num = len(encoded)
    if num > length:
        return encoding.decode(encoded[num - length:])
    else:
        return text

def scrape_text(url, length):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    response = requests.get(url, verify=False, headers=headers)
    if not isinstance(length, int):
        length = 3000
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    soup = BeautifulSoup(response.text, "html.parser")

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

class Assistant:
    def __init__(self, configs, name, instructions, model, assistant_id=None, thread_id=None, event_listener=None, openai_key=None, files=None, code_interpreter=False, retrieval=False, is_json=None, old_mode=False, max_tokens=None, bot_intro=None, get_thread=None, put_thread=None, save_memory=None, query_memory=None, max_messages=4, raw_mode=False, streaming=False, has_file=False, file_identifier=None, read_file=None, search_enabled=False, view_pages=False, search_window=1000, other_tools=None, other_functions={}):
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
        self.thread_id = thread_id
        self.old_mode = old_mode
        self.streaming = streaming
        self.has_file = has_file
        self.file_identifier = file_identifier
        self.read_file = read_file
        self.search_enabled = search_enabled
        self.view_pages = view_pages
        self.search_window = search_window
        self.other_tools = other_tools
        self.other_functions = other_functions
        if is_json is not None:
            self.is_json = is_json
        if openai_key is None:
            self.openai_client = OpenAI()
        else:
            self.openai_client = OpenAI(api_key=openai_key)
        if old_mode:
            self.assistant = None
            self.thread = None
            self.old_mode = True
            self.raw_mode = raw_mode
            if get_thread is None:
                raise ValueError("get_thread must be provided if old_mode is True")
            if put_thread is None:
                raise ValueError("put_thread must be provided if old_mode is True")
            if max_tokens is None:
                raise ValueError("max_tokens must be provided if old_mode is True")
            self.save_memory = save_memory
            self.query_memory = query_memory
            self.max_messages = max_messages
            self.get_thread = get_thread
            self.put_thread = put_thread
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

    def handle_old_mode(self, user_message, image_paths=None, user_tokens=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        print('not streaming')
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print(thread)
        
        content = user_message
        if image_paths is not None:
            for image_path in image_paths:
                #base64_image = encode_image(image_path)
                content = [{"type": "text", "text": user_message}]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_path
                    }
                })
        
        thread["messages"].append({"role": "user", "content": content, "timestamp": str(datetime.now().isoformat())})
        context = copy.deepcopy(thread["messages"][-self.max_messages:])
        additional_context = ""
        if self.query_memory is not None:
            additional_context = self.query_memory(self.thread_id, user_message, self.openai_client)
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
                data_ = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions + additional_context + desc_string}] + context,
                    "max_tokens": self.max_tokens,
                    "tools": tools,
                    "tool_choice": "auto"
                }
            else:
                data_ = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions + additional_context}] + context,
                    "max_tokens": self.max_tokens
                }
        else:
            data_ = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.instructions + additional_context}] + context,
                "max_tokens": self.max_tokens
            }
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
                completion = self.openai_client.chat.completions.create(**data_)

        response_message = completion.choices[0].message.content
        print(response_message)
        thread["messages"].append({"role": "assistant", "content": response_message})
        self.put_thread(self.thread_id, thread["messages"])
        if self.save_memory is not None:
            threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.openai_client)).start()
        return response_message

    def handle_old_mode_streaming(self, user_message, image_paths=None, user_tokens=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print('got here')
        print(thread)
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
        
        thread["messages"].append({"role": "user", "content": content})
        if len(thread["messages"]) > self.max_messages:
            thread["messages"] = thread["messages"][-self.max_messages:]
        additional_context = ""
        if self.query_memory is not None:
            additional_context = self.query_memory(self.thread_id, user_message, self.openai_client)
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
        
        data_ = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.instructions + additional_context}] + thread["messages"],
            "max_tokens": self.max_tokens,
            "stream": True,
            "tools": tools,
            "tool_choice": "auto" if len(tools) > 0 else None
        }

        print(data_)
        completion = self.openai_client.chat.completions.create(**data_)
        print(completion)

        result = ""
        tool_calls = {}
        
        for response_chunk in completion:
            delta = response_chunk.choices[0].delta
            if delta.content is not None:
                result += delta.content
                yield delta.content
            
            if delta.tool_calls:
                tool_outputs = []
                for tool_call in delta.tool_calls:
                    tool_calls[tool_call.id] = tool_call
                    try:
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
                completion = self.openai_client.chat.completions.create(**data_)

        thread["messages"].append({"role": "assistant", "content": result})
        self.put_thread(self.thread_id, thread["messages"])
        if self.save_memory is not None:
            threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": result}), self.openai_client)).start()
        return result

    def get_assistant_response(self, message, files=None, image_paths=None, user_tokens=None):
        if self.old_mode:
            if self.streaming:
                return self.handle_old_mode_streaming(message, image_paths=image_paths, user_tokens=user_tokens)
            return self.handle_old_mode(message, image_paths=image_paths, user_tokens=user_tokens)
        
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
import os
import time
import json
from dotenv import load_dotenv
import uuid
import threading

load_dotenv()

from typing import Optional



class Assistant:
    def __init__(self, configs, name, instructions, model, assistant_id=None, thread_id=None, event_listener=None, openai_key=None, files=None,code_interpreter=False, retrieval=False, is_json=None, old_mode=False, max_tokens=None, bot_intro=None, get_thread=None, put_thread=None, save_memory=None, query_memory=None, max_messages=4, raw_mode=False):
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

    # Create an OpenAI assistant and a thread for interactions
    def create_assistant_and_thread(self, files=None, code_interpreter=False, retrieval=False, bot_intro=None):
        # Extract tools from the configs
        tools = []
        model_descriptions = []
        valid_descriptions = []
        for config in self.configs:
            modified_tools = self.modify_tools_for_config(config)
            for tool in modified_tools:
                # Add 'is_json' parameter to the parameters of each tool
                """tool['function']['parameters']['properties']['is_json'] = {
                    'type': 'boolean', 
                    'description': "Do with json or not - should be used if errors with Content-Type occur. Should never be used on its own"
                }"""
                tools.append(tool)
                # Include 'is_json' in the required parameters if necessary
                # tool['function']['parameters']['required'].append('is_json')
            if config.model_description and config.model_description.lower() != "none":
                valid_descriptions.append(config.model_description)
        print(tools)
        # Concatenate valid descriptions
        if valid_descriptions:
            desc_string = " Tool information below\n---------------\n" + "\n---------------\n".join(valid_descriptions)
        else:
            desc_string = ""
        # Initialize the OpenAI assistant
        if self.assistant_id is not None:
            assistant = self.openai_client.beta.assistants.retrieve(self.assistant_id)
            if self.thread_id is not None:
                thread = self.openai_client.beta.threads.retrieve(self.thread_id)
                runs = self.openai_client.beta.threads.runs.list(self.thread_id)
                if len(runs.data) > 0:
                    latest_run = runs.data[0]
                    if(latest_run.status == "in_progress" or latest_run.status == "queued" or latest_run.status == "requires_action"):
                        run = self.openai_client.beta.threads.runs.cancel(thread_id=self.thread_id, run_id = latest_run.id)
                        print('cancelled run')
            else:
                thread = None
                if bot_intro is not None:
                    thread = self.openai_client.beta.threads.create(messages=[{"role": "user", "content": "Before the thread, you said "+bot_intro}])
                else:
                    thread = self.openai_client.beta.threads.create()
        else:
            file_ids = None
            if files is not None:
                file_ids = []
                for file in files:
                    file = self.openai_client.create(
                        file = open(file, 'rb'),
                        purpose='assistants'
                    )
                    file_ids.append(file.id)
            if code_interpreter:
                tools.append({"type": "code_interpreter"})
            if retrieval:
                tools.append({"type": "retrieval"})
            assistant = None
            if file_ids is not None:
                assistant = self.openai_client.beta.assistants.create(
                name=self.name,
                instructions=self.instructions+desc_string,
                model=self.model,
                tools=tools,
                file_ids=file_ids if file_ids is not None else None 
            )
            else:
                assistant = self.openai_client.beta.assistants.create(
                    name=self.name,
                    instructions=self.instructions+desc_string,
                    model=self.model,
                    tools=tools,
                )
            self.assistant_id = assistant.id
            thread = None
            if bot_intro is not None:
                thread = self.openai_client.beta.threads.create(messages=[{"role": "user", "content": "Before the thread, you said "+bot_intro}])
            else:
                thread = self.openai_client.beta.threads.create()
            self.thread_id = thread.id
            #print("Thread ID: save this for persistence: "+thread.id)

        # Create a thread for the assistant
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
    def handle_old_mode(self, user_message, user_tokens=None):
        if self.thread_id is None:
            self.thread_id = str(uuid.uuid4())

        # Get the current thread
        thread = self.get_thread(self.thread_id)
        if thread is None:
            thread = {"messages": []}
        print(thread)
        thread["messages"].append({"role": "user", "content": user_message})
        if len(thread["messages"]) > self.max_messages:
            thread["messages"] = thread["messages"][-self.max_messages:]
        additional_context = ""
        if self.query_memory is not None:
            additional_context = self.query_memory(self.thread_id, user_message,self.openai_client)
        if additional_context is None:
            additional_context ="\nInformation from the past that may be relevant: "+additional_context
        tools = []
        model_descriptions = []
        valid_descriptions = []
        data_ = {}
        if self.raw_mode is False:
            for config in self.configs:
                modified_tools = self.modify_tools_for_config(config)
                for tool in modified_tools:
                    # Add 'is_json' parameter to the parameters of each tool
                    """tool['function']['parameters']['properties']['is_json'] = {
                        'type': 'boolean', 
                        'description': "Do with json or not - should be used if errors with Content-Type occur. Should never be used on its own"
                    }"""
                    tools.append(tool)
                    # Include 'is_json' in the required parameters if necessary
                    # tool['function']['parameters']['required'].append('is_json')
                if config.model_description and config.model_description.lower() != "none":
                    valid_descriptions.append(config.model_description)
            desc_string = ""
            if valid_descriptions:
                desc_string = " Tool information below\n---------------\n" + "\n---------------\n".join(valid_descriptions)
            else:
                desc_string = ""
            if len(tools) > 0:
                data_ = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions+additional_context+desc_string}] + thread["messages"],
                    "max_tokens": self.max_tokens,
                    "tools": tools,
                    "tool_choice": "auto"
                }
            else:
                data_ = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.instructions+additional_context}] + thread["messages"],
                    "max_tokens": self.max_tokens
                }
        else: 
            data_ = {
                "model": self.model,
                "messages": [{"role": "system", "content": self.instructions+additional_context}] + thread["messages"],
                "max_tokens": self.max_tokens
            }
        completion = self.openai_client.chat.completions.create(**data_)
        if self.raw_mode == False:
            while completion.choices[0].message.role == "assistant" and completion.choices[0].message.tool_calls:
                tool_outputs = []
                for tool_call in completion.choices[0].message.tool_calls:
                    # Execute the function associated with the tool
                    result = self.execute_function(tool_call.function.name, tool_call.function.arguments, user_tokens)
                    output = {
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(result)
                    }
                    tool_outputs.append(output)
                    self.event_listener(output)

                # Resend the completion request with the tool outputs
                data_['tool_outputs'] = tool_outputs
                completion = self.openai_client.ChatCompletion.create(**data_)
                

        # Extract the response from the completion
        response_message = completion.choices[0].message.content

        # Add the response to the thread
        thread["messages"].append({"role": "assistant", "content": response_message})

        # Save the updated thread
        self.put_thread(self.thread_id, thread["messages"])

        # If save_memory is not None, use it to store the input and output
        if self.save_memory is not None:
            #use threading to save memory
            threading.Thread(target=self.save_memory, args=(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.openai_client)).start()
            #self.save_memory(self.thread_id, json.dumps({"input": user_message, "output": response_message}), self.openai_client)

        return response_message
    def get_assistant_response(self,message, user_tokens=None):
        if self.old_mode:
            return self.handle_old_mode(message, user_tokens=user_tokens)
        message = self.openai_client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=message
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
                    #print(tool_call)
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
                        #print(output)
                        #put output to event listener if there is one
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
    def execute_function(self,function_name, arguments, user_token=None):
        """Execute a function and return the result."""
        #example of function_name: "alpha_vantage/query"
        #config.make_api_call_by_operation_id("genericQuery", params={"function": "TIME_SERIES_DAILY", "symbol": "BTC", "market": "USD"}
        #config.make_api_call_by_path("/query", "GET", params={"function": "TIME_SERIES_DAILY", "symbol": "BTC", "market": "USD"})
        #actual implementation of the function
        #turn arguments into dictionary
        print(function_name)
        print(arguments)
        if self.multiple_configs:
            config_name, actual_function_name = function_name.split('-', 1)
            config = next((cfg for cfg in self.configs if cfg.name == config_name), None)
        else:
            actual_function_name = function_name
            config = self.configs[0]

        if not config:
            return "Configuration not found for function: " + function_name

        arguments = json.loads(arguments)
        is_json = config.is_json
        
        try:
            request = config.make_api_call_by_operation_id(actual_function_name, params=arguments, is_json=is_json, user_token=user_token)
            print(request)
            print(request.status_code)
            print(request.reason)
            try:
                return request.json()+"\n "+str(request.status_code)+" "+request.reason
            except Exception as e:
                return request.text+"\n "+str(request.status_code)+" "+request.reason
        except Exception as e:
            print(e)
            try:
                #split the function name into path and method by - eg query-GET
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
                #check if response is json
                try:
                    return request.json()+"\n "+str(request.status_code)+" "+request.reason
                except Exception as e:
                    return request.text+"\n "+str(request.status_code)+" "+request.reason
            except Exception as e:
                print(e)
                #debug stack trace
                import traceback
                traceback.print_exc()
                try:
                    request = config.make_api_call_by_path('/'+path, method.upper(), params=arguments, is_json=is_json, user_token=user_token)
                    print(request)
                    
                    #check if json
                    print(request.text)
                    print(request.status_code)
                    print(request.reason)
                    try:
                        return request.json()+"\n "+str(request.status_code)+" "+request.reason
                    except Exception as e:
                        return request.text+"\n "+str(request.status_code)+" "+request.reason
                except Exception as e:
                    print(e)
                    return "Error"

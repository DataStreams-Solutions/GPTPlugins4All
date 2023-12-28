import os
import time
import json
from dotenv import load_dotenv

load_dotenv()

class Assistant:
    def __init__(self, configs, name, instructions, model, assistant_id=None, thread_id=None, event_listener=None, openai_key=None, files=None,code_interpreter=False, retrieval=False, is_json=None):
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
        if is_json is not None:
            self.is_json = is_json
        if openai_key is None:
            self.openai_client = OpenAI()
        else:
            self.openai_client = OpenAI(api_key=openai_key)
        self.assistant, self.thread = self.create_assistant_and_thread(files=files, code_interpreter=code_interpreter, retrieval=retrieval)

    # Create an OpenAI assistant and a thread for interactions
    def create_assistant_and_thread(self, files=None, code_interpreter=False, retrieval=False):
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
    def get_assistant_response(self,message, user_tokens=None):
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
            time.sleep(5)
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

import uuid
import json
import yaml
from enum import Enum
import requests
from openapi_spec_validator import openapi_v3_spec_validator  # You may need to install this package
from urllib.parse import urlencode
import logging
import os
logging.basicConfig(level=logging.INFO)

class AuthMethod(Enum):
    OAUTH = 'OAUTH'
    QUERY = 'QUERY'
    BASIC = 'BASIC'
    HEADER = 'HEADER'
    NONE = 'NONE'

class Config:
    #cache_file = 'search_cache.json'
    #cache file should be in hidden folder
    #make folder if it doesn't exist
    
    
    
    def __init__(self, input_value, api_key=None, validate=False, name=None, use_cache=True):
        self.spec_string = None
        self.model_description = None
        self.is_json = None
        if use_cache:
            cache_dir = os.path.join(os.path.expanduser('~'), '.gpt-plugins-4all')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            cache_file = os.path.join(cache_dir, 'search_cache.json')
            self.use_cache = True
        if self.is_valid_spec_string(input_value):
            # Input is a spec string
            self.spec_string = input_value
            self.name = name
        else:
            # Input is assumed to be a config name
            self.spec_string = self.fetch_spec_from_name(input_value, api_key)
            self.model_description = self.get_model_description_by_name(input_value, api_key)
            self.name = input_value

        self.spec_object = None
        self.auth_methods = {}
        self.id = str(uuid.uuid4())
        
        # Validate and parse the OpenAPI spec
        if validate:
            self.validate_and_parse_spec(self.spec_string)
        else:
            self.parse_no_validate(self.spec_string)

    
    def is_valid_spec_string(self, input_value):
        # Implement logic to check if the input value is a valid spec string
        # This can be a simple check like whether it starts with '{' or is a YAML format
        try:
            self.validate_and_parse_spec(input_value)
            return True
        except:
            return False
    @staticmethod
    def fetch_spec_from_name(config_name, api_key=None):
        # Use the existing method to fetch the config by name
        config_data = Config.fetch_spec_by_name(config_name, api_key)
        # Extract the spec string from the config data
        return config_data
    @classmethod
    def from_existing_config(cls, existing_config):
      """Create a Config object from an existing configuration dictionary."""
      required_fields = ['spec_string', 'spec_id', 'auth_methods']
      if not all(field in existing_config for field in required_fields):
          raise ValueError("Existing configuration is missing required fields")

      return cls(
          spec_string=existing_config['spec'],
          spec_id=existing_config['id'],
          auth_methods=existing_config['auth_methods']
      )
    @staticmethod
    def load_cache():
        if Config.use_cache and os.path.exists(Config.cache_file):
            with open(Config.cache_file, 'r') as file:
                return json.load(file)
        return []

    @staticmethod
    def save_cache(cache_data):
        if Config.use_cache:
            with open(Config.cache_file, 'w') as file:
                json.dump(cache_data, file, indent=4)
    @staticmethod
    def fetch_user_configs(api_key):
        if not api_key:
            raise ValueError("API key is required to fetch user configs")
        headers = {'Authorization': f'Bearer {api_key}'}
        response = requests.get('https://api.gptplugins4all.com/user/configs', headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch user configs: {response.status_code} - {response.text}")
    @staticmethod
    def create_config_from_url(api_key, url, auth_type='none', visibility='private'):
        if not api_key:
            raise ValueError("API key is required to create a new config")
        headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
        payload = json.dumps({
            "url": url,
            "auth_type": auth_type,
            "visibility": visibility
        })
        response = requests.post('https://api.gptplugins4all.com/configs/create_from_plugin_json_url', headers=headers, data=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to create config from URL: {response.status_code} - {response.text}")
    @staticmethod
    def search_configs(query, api_key=None, use_cache=True):
        cached_configs = Config.load_cache() if use_cache else []

        if use_cache:
            # Search within the cached data
            return [config for config in cached_configs if query.lower() in config.get('name', '').lower() or query.lower() in config.get('description', '').lower()]

        # If not using cache, fetch new data and update the cache
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(
                'http://localhost:5000/configs',
            params={'query': query},
            headers=headers
        )

        if response.status_code == 200:
            new_configs = response.json()
            # Deduplicate and update cache
            existing_ids = {config['config_id'] for config in cached_configs}
            for config in new_configs:
                if config['config_id'] not in existing_ids:
                    cached_configs.append(config)
            Config.save_cache(cached_configs)
            return new_configs
        else:
            raise Exception(f"Search failed: {response.status_code} - {response.text}")
    @staticmethod
    def fetch_config_by_id_or_name(config_id, api_key=None):
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(
            f'https://api.gptplugins4all.com/configs/{config_id}/fetch',
            headers=headers
        )
        if response.status_code == 200:
            config_data = response.json()
            return config_data
        else:
            raise Exception(f"Fetch failed: {response.status_code} - {response.text}")
    @staticmethod
    def fetch_and_cache(query, api_key, cache, cache_key):
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(
            'https://api.gptplugins4all.com/configs',
            params={'query': query},
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()
            cache[cache_key] = result
            Config.save_cache(cache)
            return result
        else:
            raise Exception(f"Search failed: {response.status_code} - {response.text}")
   

    def validate_and_parse_spec(self, spec_string):
      try:
          spec_object = yaml.safe_load(spec_string) if self.is_yaml(spec_string) else json.loads(spec_string)

          # Validate the spec object against OpenAPI 3.1.0 standards
          errors_iterator = openapi_v3_spec_validator.iter_errors(spec_object)
          errors = list(errors_iterator)
          if errors:
              raise ValueError(f"Spec validation errors: {errors}")

          self.spec_object = spec_object
      except Exception as e:
          raise ValueError(f"Error loading OpenAPI spec: {e}")
    def parse_no_validate(self, spec_string):
        try:
            spec_object = yaml.safe_load(spec_string) if self.is_yaml(spec_string) else json.loads(spec_string)
            self.spec_object = spec_object
        except Exception as e:
            raise ValueError(f"Error loading OpenAPI spec: {e}")

    @staticmethod
    def is_yaml(string):
        try:
            yaml.safe_load(string)
            return True
        except yaml.YAMLError:
            return False

    @classmethod
    def import_from_name(cls, name, api_key=None):
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        response = requests.get(
            f'https://gptplugins4all.com/configs/{name}/fetch',
            headers=headers
        )
        if response.status_code == 200:
            config_data = response.json()
            return cls(spec_string=json.dumps(config_data))
        else:
            raise Exception(f"Import failed: {response.status_code} - {response.text}")
    # Method to add authentication method
    def add_auth_method(self, method_name, method_details):
        if method_name not in AuthMethod.__members__:
            raise ValueError(f"Invalid authentication method: {method_name}")
        self.auth_methods['method'] = method_name
        self.auth_methods['details'] = method_details

    def start_oauth_flow(self):
      """Initiates the OAuth flow."""
      auth_config = self.auth_methods.get('details')
      if not auth_config:
          raise ValueError("OAuth configuration not found")

      params = {
          "response_type": auth_config["response_type"],
          "client_id": auth_config["client_id"],
          "redirect_uri": auth_config["redirect_uri"],
          "scope": auth_config.get("scope", "")
      }
      if 'custom_parameters' in auth_config:
        params.update(auth_config['custom_parameters'])
      auth_url = f"{auth_config['auth_url']}?{urlencode(params)}"
      return auth_url
    def handle_oauth_callback(self, code):
      """Handles the OAuth callback and exchanges the code for tokens."""
      auth_config = self.auth_methods.get('details')
      if not auth_config or not code:
          raise ValueError("OAuth configuration not found or no code provided")

      token_data = {
          "grant_type": "authorization_code",
          "code": code,
          "redirect_uri": auth_config["redirect_uri"],
          "client_id": auth_config["client_id"],
          "client_secret": auth_config["client_secret"]
      }
      response = requests.post(auth_config["token_url"], data=token_data)

      if response.status_code == 200:
          print(response.json())
          print('got token')
          return response.json()
      else:
          raise Exception(f"Failed to obtain tokens: {response.status_code}")
    #creates a simplified representation for the AI to understand
    def generate_simplified_api_representation(self):
      info = self.spec_object.get('info', {})
      simplified_api_representation = (
          f"Title: {info.get('title', 'No title')}\n"
          f"Version: {info.get('version', 'No version')}\n"
          f"Description: {info.get('description', 'No description')}\n\n"
      )
      for path, methods in self.spec_object.get("paths", {}).items():
          for method, details in methods.items():
              operation_id = details.get('operationId', 'Unnamed')
              simplified_api_representation += f"Name: {operation_id}\n"
              simplified_api_representation += f"Endpoint: {path} {method.upper()}\n"
              simplified_api_representation += f"   Description: {details.get('summary', details.get('description', 'No description'))}\n"
              simplified_api_representation += "   Parameters: "
              
              # Add parameters info
              for param in details.get("parameters", []):
                  simplified_api_representation += f"{param['name']} ({param['in']} parameter)"
                  if param.get('required'):
                      simplified_api_representation += " [required]"
                  simplified_api_representation += ", "
              
              if 'requestBody' in details:
                  simplified_api_representation += "Body parameters"

              simplified_api_representation = simplified_api_representation.strip(", ")
              simplified_api_representation += "\n   Response: ...\n\n"
        
      return simplified_api_representation
    # Method to get base url
    def get_base_url(self):
        # Fetch the first server URL from spec_object
        servers = self.spec_object.get('servers', [])
        if servers:
            return servers[0].get('url', '')
        return ''
    # Function to make API call
    def make_api_call_by_operation_id(self, operation_id, params, user_token=None, is_json=False):
        endpoint, method = self.find_endpoint_by_operation_id(operation_id)
        if not endpoint:
            raise ValueError(f"OperationId '{operation_id}' not found in API spec")
        base_url = self.get_base_url()
        # Replace placeholders in the endpoint with actual parameter values
        endpoint = endpoint.format(**params)
        url = f"{base_url}{endpoint}"
        if method.upper() == 'GET' or method.upper() == 'DELETE':
            is_json = False
        headers, params = self.prepare_auth(user_token, params)
        
        if is_json:
            headers['Content-Type'] = 'application/json'
            response = getattr(requests, method.lower())(url, json=params, headers=headers)
        else:
            response = getattr(requests, method.lower())(url, params=params, headers=headers)
        
        return response
    
    def prepare_auth(self, user_token, params):
        headers = {}
        auth_method = self.auth_methods.get('method', AuthMethod.NONE.value)
        auth_details = self.auth_methods.get('details', {})
        if auth_method == AuthMethod.OAUTH.value and user_token:
            headers["Authorization"] = f"Bearer {user_token['access_token']}"
        elif auth_method == AuthMethod.BASIC.value:
            headers['Authorization'] = f"Basic {auth_details['key']}"
        elif auth_method == AuthMethod.HEADER.value:
            headers[auth_details['header_name']] = auth_details['key']
        elif auth_method == AuthMethod.QUERY.value:
            params[auth_details['param_name']] = auth_details['key']
        return headers, params
    def make_api_call_by_path(self, path, method, params, user_token=None, is_json=False):
        base_url = self.get_base_url()
        # Replace placeholders in the path with actual parameter values
        path = path.format(**params)
        url = f"{base_url}{path}"
        headers, params = self.prepare_auth(user_token, params)
        if is_json:
            headers['Content-Type'] = 'application/json'
            response = getattr(requests, method.lower())(url, json=params, headers=headers)
        else:
            response = getattr(requests, method.lower())(url, params=params, headers=headers)
        return response
    def find_endpoint_by_operation_id(self, operation_id):
        for path, methods in self.spec_object.get("paths", {}).items():
            for method, details in methods.items():
                if details.get('operationId') == operation_id:
                    return path, method
        return None, None
    def get_ref(self, ref):
        parts = ref.split("/")
        obj = self.spec_object
        for part in parts:
            if part == '#':
                continue
            if part not in obj:
                raise ValueError(f"Reference {ref} not found in schema")
            obj = obj[part]
        return obj
    def resolve_ref(self, obj):
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref_obj = self.get_ref(obj['$ref'])
                return self.resolve_ref(ref_obj)
            else:
                return {k: self.resolve_ref(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.resolve_ref(item) for item in obj]
        else:
            return obj


    def extract_request_body(self, request_body):
        if 'content' in request_body and 'application/json' in request_body['content']:
            schema = request_body['content']['application/json'].get('schema', {})
            return self.resolve_ref(schema)  # Return the entire resolved schema
        return {}
    def generate_tools_representation(self):
        tools = []

        for path, methods in self.spec_object.get("paths", {}).items():
            for method, details in methods.items():
                tool_parameters = self.extract_parameters(details.get("parameters", []))

                if 'requestBody' in details:
                    body_params = self.extract_request_body(details['requestBody'])
                    # Merge the requestBody properties directly into the parameters properties
                    if 'properties' in body_params:
                        tool_parameters['properties'].update(body_params['properties'])

                tool = {
                    "type": "function",
                    "function": {
                        "name": details.get('operationId', path + "-" + method),
                        "description": details.get('description', 'No description'),
                        "parameters": tool_parameters
                    }
                }
                tools.append(tool)

        return tools
    def extract_parameters(self, parameters):
        params = {"type": "object", "properties": {}, "required": []}
        for param in parameters:
            if param['in'] == 'path':
                param_name = param['name']
                param_details = param.get('schema', {})
                resolved_param_details = self.resolve_ref(param_details)
                params["properties"][param_name] = resolved_param_details
                if param.get('required', False):
                    params["required"].append(param_name)
        return params
    @staticmethod 
    def fetch_spec_by_name(config_name, api_key=None):
        url = f"https://api.gptplugins4all.com/configs/{config_name}/fetch"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            config_data = response.json()
            return config_data['spec']
        else:
            raise Exception(f"Error fetching config: {response.content}")
    @staticmethod
    def get_model_description_by_name(config_name, api_key=None):
        url = f"https://api.gptplugins4all.com/configs/{config_name}/fetch"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            config_data = response.json()
            return config_data.get('model_description', 'none')
        else:
            raise Exception(f"Error fetching config: {response.content}")
    @staticmethod
    def fetch_and_load_config_by_name(config_name, api_key=None):
        url = f"https://api.gptplugins4all.com/configs/{config_name}/fetch"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            config_data = response.json()
            print(config_data)
            print(config_data['spec'])
            return Config(config_data['spec'])
        else:
            raise Exception(f"Error fetching config: {response.content}")

    
   

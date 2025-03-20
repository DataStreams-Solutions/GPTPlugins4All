# GPT Plugins 4All <img src="https://www.gptplugins4all.com/gptplugins4all.webp" alt="GPTPlugins4All logo" width="70"/>
GPT Plugins 4All is a Python library designed to facilitate the integration of GPT and other large language models with various APIs, leveraging OpenAPI specifications. This library simplifies the process of parsing OpenAPI specs, managing different authentication methods, and dynamically interacting with APIs based on model responses.
[![PyPI version](https://badge.fury.io/py/GPTPlugins4All.svg)](https://badge.fury.io/py/GPTPlugins4All)
![Demo using the AlphaVantage API with OpenAI](https://github.com/tcmartin/GPTPlugins4All/blob/master/demo/demo.gif)

## Features

- Parse and validate OpenAPI 3.1.0 specifications.
- Handle diverse authentication methods, including OAuth 2.0, Basic Auth, Header Auth, and Query Parameter Auth.
- Generate structured API representations for AI interactions.
- Dynamically construct API calls based on OpenAPI specs.
- Support OAuth2.0 flow for token acquisition and usage.
- Easily create and manage instances of AI assistants and threads for interactive sessions
- Command-Line Interface (CLI) for convenient management of configurations and interactions.
- Turn specs into function calls for AI
- Import specs by name from repo of over 200 plugins
- Easily create AI Assistants that have these functions (and search, view page, and RAG, and others) using any openai-compatible api
- AI assistant streaming for real-time applications like phone calls
- Handles images, sends events
## Installation

Install GPT Plugins 4All using pip:

```bash
pip install GPTPlugins4All
```
## Using the CLI
The GPT Plugins 4All CLI provides a convenient way to manage configurations and interact with your APIs from the command line.
###Common Commands
Search for Configurations
```bash
gpt-plugins-4all search --query "your_search_query"
```
Fetch a Specific Configuration
```bash
gpt-plugins-4all get --id "config_id_or_name"
```
List Your Configurations
```bash
gpt-plugins-4all my-configs --api-key "your_api_key"
```
Submit a New Configuration
```bash
gpt-plugins-4all submit-config --url "config_url" --auth-type "auth_type" --visibility "visibility" --api-key "your_api_key"
```
## Usage
The CLI supports various operations such as searching for configurations, retrieving specific configurations, listing user configurations, and submitting new configurations. You can use these commands directly from your terminal to interact with the GPT Plugins 4All library.

For detailed usage and available options for each command, use the --help flag with any command:
```bash
gptplugins4all [command] --help
```

## Quick Start

### Initializing with an OpenAPI Specification
We support initializing with an OpenAPI Spec in two ways. One way is to just give the name of the spec from [https://gptplugins4all.com](https://gptplugins4all.com) like this:
```python
config = Config('alpha_vantage')
``` 
We also support directly making a config from an OpenAPI spec.
```python
from GPTPlugins4All.config import Config

# Initialize the Config object with your OpenAPI spec
spec_string = """..."""  # Your OpenAPI spec as a string
config = Config(spec_string)
```

### Adding Authentication Methods

#### Add Basic Authentication

```python
config.add_auth_method("BASIC", {"key": "your_api_key"})
```

#### Add OAuth Configuration

```python
config.add_auth_method("OAUTH", {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "auth_url": "https://example.com/auth",
    "token_url": "https://example.com/token",
    "redirect_uri": "https://yourapp.com/oauth-callback",
    "scope": "read write"
})
```

### Generating Simplified API Representations

```python
simplified_api = config.generate_simplified_api_representation()
print(simplified_api)
```
### Generate Object for use with OpenAI functions
```python
tools = config.generate_tools_representation()
```
### Using the Assistant Class
The Assistant class (for now) provides a simplified interface between your plugins and various OpenAI models via the Assistants API.

Initializing the Assistant
```python
from assistant import Assistant

# Create an assistant instance
my_assistant = Assistant(config, "My Assistant", "Your instructions", "model_name")
```
Interacting with the assistant
```python
# Getting a response from the assistant
response = my_assistant.get_assistant_response("Your query here")
print(response)
```
### OAuth Flow

```python
auth_url = config.start_oauth_flow()
# Redirect the user to auth_url...

tokens = config.handle_oauth_callback(code_from_redirect)
```

### Making API Calls

```python
response = config.make_api_call("/endpoint", "GET", {"param": "value"})
```

#### Oauth
```python
url = config5.start_oauth_flow() #use this url to get code first
callback = config5.handle_oauth_callback(code)
#example
response = config5.make_api_call_by_path(path, "POST", params=your_params, user_token=callback, is_json=True)
```

## Contributing

Contributions are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md).

## License

GPT Plugins 4All is released under the [MIT License](LICENSE).

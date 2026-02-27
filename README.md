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
- Easily create AI Assistants that have these functions (and search, view page, and RAG, and others) using any openai-compatible api (eg OpenRouter, novita, Deepseek)
- AI assistant streaming for real-time applications like phone calls
- Handles images, sends events
- Built-in web search tool (`search_google`) now supports Serper (`SERPER_API_KEY`) with automatic fallback to free DuckDuckGo HTML search

### Web Search Provider

`search_google` (tool name kept for backward compatibility) now uses this provider order:

1. Serper (`SERPER_API_KEY`) if configured
2. DuckDuckGo HTML search (free, no API key)
3. Legacy `googlesearch` package if installed

Optional overrides:

- `SERPER_API_KEY`: Enables Serper for Google-quality results
- `SERPER_API_URL`: Custom Serper-compatible endpoint (default `https://google.serper.dev/search`)
- `GPTPLUGINS_SEARCH_PROVIDER`: Force preferred order start (`serper`, `duckduckgo`, or `google`)
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

## MCP (Model Context Protocol) Integration

GPTPlugins4All now supports MCP integration, allowing you to connect to external MCP servers and use their tools seamlessly within your AI assistants.

### What is MCP?

MCP (Model Context Protocol) is a standard for connecting AI assistants to external tools and data sources. It allows you to extend your assistant's capabilities with specialized tools like web scraping, browser automation, file operations, and more.

### Prerequisites

#### 1. Install MCP Library
```bash
pip install mcp
```

#### 2. Install Node.js and npm
MCP servers often run as Node.js applications. Install Node.js from [nodejs.org](https://nodejs.org/) or using a package manager:

```bash
# macOS with Homebrew
brew install node

# Ubuntu/Debian
sudo apt install nodejs npm

# Windows with Chocolatey
choco install nodejs
```

#### 3. Install MCP Servers

Install the MCP servers you want to use:

```bash
# Playwright MCP (browser automation)
npm install -g @playwright/mcp

# Firecrawl MCP (advanced web scraping)
npm install -g firecrawl-mcp

# Other popular MCP servers
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-sqlite
```

### Setting Up MCP with GPTPlugins4All

#### Basic MCP Configuration

```python
from GPTPlugins4All.assistant import Assistant

# Configure MCP servers
mcp_servers = {
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest"]
    },
    "firecrawl": {
        "command": "npx", 
        "args": ["-y", "firecrawl-mcp"],
        "env": {
            "FIRECRAWL_API_URL": "https://api.firecrawl.dev"  # Optional: custom Firecrawl instance
        }
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"]
    }
}

# Create assistant with MCP capabilities
assistant = Assistant(
    configs=[],  # Your existing API configs
    name="MCP-Enabled Assistant",
    instructions="You have access to web scraping, browser automation, and file system tools.",
    model="gpt-4o",
    old_mode=True,
    max_tokens=2000,
    mcp_servers=mcp_servers  # Add MCP servers here
)
```

#### Available MCP Tools

Once configured, your assistant automatically gains access to all tools from the connected MCP servers:

**Playwright Tools (Browser Automation):**
- `mcp_playwright_browser_navigate` - Navigate to URLs
- `mcp_playwright_browser_click` - Click elements
- `mcp_playwright_browser_type` - Type text
- `mcp_playwright_browser_screenshot` - Take screenshots
- `mcp_playwright_browser_fill_form` - Fill forms
- And 15+ more browser automation tools

**Firecrawl Tools (Advanced Web Scraping):**
- `mcp_firecrawl_scrape` - Scrape single pages with advanced options
- `mcp_firecrawl_crawl` - Crawl entire websites
- `mcp_firecrawl_search` - Search the web and extract content
- `mcp_firecrawl_map` - Discover all URLs on a website
- `mcp_firecrawl_extract` - Extract structured data using AI

### Complete Example

```python
from GPTPlugins4All.assistant import Assistant

# MCP server configuration
mcp_servers = {
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest"]
    },
    "firecrawl": {
        "command": "npx",
        "args": ["-y", "firecrawl-mcp"]
    }
}

# Create assistant
assistant = Assistant(
    configs=[],
    name="Web Research Assistant",
    instructions="""You are a web research assistant with powerful scraping and browser automation capabilities.
    
    You can:
    - Navigate websites and take screenshots
    - Fill out forms and interact with web pages
    - Scrape content from any website
    - Search the web and extract relevant information
    - Crawl entire websites for comprehensive data
    - Extract structured data from web pages
    
    Always use the most appropriate tool for each task.""",
    model="gpt-4o",
    old_mode=True,
    max_tokens=3000,
    mcp_servers=mcp_servers
)

# The assistant now has access to all MCP tools automatically!
# You can interact with it normally, and it will use MCP tools as needed
response = assistant.handle_old_mode("Please scrape the latest news from example.com and take a screenshot")
```

### MCP Server Configuration Options

Each MCP server configuration supports these options:

```python
{
    "server_name": {
        "command": "npx",           # Command to run the server
        "args": ["server-package"], # Arguments for the command
        "env": {                    # Environment variables
            "API_KEY": "your-key",
            "CONFIG_OPTION": "value"
        }
    }
}
```

### Popular MCP Servers

| Server | Package | Description |
|--------|---------|-------------|
| Playwright | `@playwright/mcp` | Browser automation and web interaction |
| Firecrawl | `firecrawl-mcp` | Advanced web scraping and crawling |
| Filesystem | `@modelcontextprotocol/server-filesystem` | File system operations |
| SQLite | `@modelcontextprotocol/server-sqlite` | Database operations |
| GitHub | `@modelcontextprotocol/server-github` | GitHub API integration |
| Slack | `@modelcontextprotocol/server-slack` | Slack API integration |

### Troubleshooting MCP Setup

#### Common Issues

1. **"MCP library not available"**
   ```bash
   pip install mcp
   ```

2. **"Command not found: npx"**
   ```bash
   # Install Node.js and npm first
   npm install -g npx
   ```

3. **MCP server fails to start**
   - Check that the server package is installed globally
   - Verify Node.js version compatibility
   - Check server-specific requirements

4. **No tools loaded from MCP server**
   - Verify server configuration is correct
   - Check server logs for errors
   - Ensure required environment variables are set

#### Testing MCP Integration

Run the test suite to verify MCP integration:

```bash
python test_suite.py mcp
```

This will test connections to Playwright and Firecrawl MCP servers and verify that tools are loaded correctly.

### Advanced MCP Usage

#### Custom MCP Server Configuration

You can create custom MCP server configurations for specialized use cases:

```python
mcp_servers = {
    "custom_scraper": {
        "command": "python",
        "args": ["/path/to/your/mcp_server.py"],
        "env": {
            "CUSTOM_API_KEY": "your-api-key",
            "DEBUG": "true"
        }
    }
}
```

#### Conditional MCP Loading

Load MCP servers conditionally based on environment or requirements:

```python
import os

mcp_servers = {}

# Only load Playwright if browser automation is needed
if os.getenv("ENABLE_BROWSER_AUTOMATION"):
    mcp_servers["playwright"] = {
        "command": "npx",
        "args": ["@playwright/mcp@latest"]
    }

# Only load Firecrawl if API key is available
if os.getenv("FIRECRAWL_API_KEY"):
    mcp_servers["firecrawl"] = {
        "command": "npx",
        "args": ["-y", "firecrawl-mcp"],
        "env": {"FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")}
    }
```

## Contributing

Contributions are welcome! Please check out the [contributing guidelines](CONTRIBUTING.md).

## License

GPT Plugins 4All is released under the [MIT License](LICENSE).

# GPTPlugins4All Testing Suite

This directory contains comprehensive tests for the GPTPlugins4All library, covering all major functionality including MCP integration, web scraping, and assistant capabilities.

## Quick Start

Run all tests:
```bash
python test_suite.py
```

Run a specific test:
```bash
python test_suite.py mcp          # Test MCP integration
python test_suite.py scraping     # Test basic web scraping
python test_suite.py javascript   # Test JavaScript-heavy site scraping
python test_suite.py assistant    # Test basic assistant functionality
python test_suite.py gpt5         # Test GPT-5 token handling
```

## Test Categories

### 1. Basic Web Scraping
Tests the `scrape_text` function with simple HTML pages.
- **Purpose**: Verify basic HTTP request and HTML parsing
- **Test URL**: httpbin.org/html
- **Expected**: Successfully extract text content

### 2. JavaScript Scraping (Playwright Fallback)
Tests scraping of JavaScript-heavy websites using Playwright fallback.
- **Purpose**: Verify headless browser fallback works for dynamic content
- **Test URL**: versabot.co
- **Expected**: Extract meaningful content from JS-rendered page

### 3. MCP Integration
Tests Model Context Protocol integration with external servers.
- **Purpose**: Verify MCP client can connect to and use external tools
- **Servers Tested**: 
  - Playwright MCP (browser automation)
  - Firecrawl MCP (web scraping)
- **Expected**: Successfully connect and load tools from both servers

### 4. Basic Assistant Functionality
Tests core Assistant class functionality without external dependencies.
- **Purpose**: Verify assistant initialization and configuration
- **Features Tested**: Model setup, tool configuration, search/scraping enablement
- **Expected**: Assistant creates successfully with correct settings

### 5. GPT-5 Token Handling
Tests the new GPT-5 token parameter handling.
- **Purpose**: Verify `max_completion_tokens` vs `max_tokens` logic
- **Models Tested**: GPT-5 (uses max_completion_tokens), GPT-4 (uses max_tokens)
- **Expected**: Correct token parameter based on model type

## Prerequisites

### Required Dependencies
```bash
pip install requests beautifulsoup4 playwright mcp
playwright install chromium
```

### Optional Dependencies (for MCP tests)
```bash
# For Playwright MCP server
npm install -g @playwright/mcp

# For Firecrawl MCP server  
npm install -g firecrawl-mcp
```

## MCP Integration Example

The test suite includes a complete example of MCP integration:

```python
from assistant import Assistant

# Configure MCP servers
mcp_servers = {
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest"]
    },
    "firecrawl-mcp": {
        "command": "npx", 
        "args": ["-y", "firecrawl-mcp"],
        "env": {"FIRECRAWL_API_URL": "https://firecrawl.versabot.co"}
    }
}

# Create assistant with MCP capabilities
assistant = Assistant(
    configs=[],
    name="MCP Assistant",
    instructions="You have access to web scraping and browser automation tools through MCP servers.",
    model="gpt-4o",
    old_mode=True,
    max_tokens=2000,
    mcp_servers=mcp_servers
)

# Assistant now has access to all MCP tools automatically
# Available tools include:
# - mcp_firecrawl-mcp_firecrawl_scrape (web scraping)
# - mcp_playwright_browser_navigate (browser automation)
# - mcp_playwright_browser_screenshot (screenshots)
# - And many more...
```

## Expected Output

When all tests pass, you should see:
```
🧪 GPTPlugins4All Testing Suite
============================================================
Starting comprehensive test suite...

[Individual test outputs...]

============================================================
TEST SUMMARY
============================================================
✓ PASSED   Basic Web Scraping
✓ PASSED   JavaScript Scraping
✓ PASSED   MCP Integration
✓ PASSED   Basic Assistant Functionality
✓ PASSED   GPT-5 Token Handling

Results: 5/5 tests passed
Duration: 15.23 seconds
🎉 All tests passed!
```

## Troubleshooting

### Common Issues

1. **Playwright not installed**: Run `playwright install chromium`
2. **MCP servers not available**: Install with `npm install -g @playwright/mcp firecrawl-mcp`
3. **Network timeouts**: Some tests may fail due to network connectivity
4. **Missing dependencies**: Install all required packages with pip

### Test Failures

If tests fail, check:
- Internet connectivity for scraping tests
- Node.js and npm installation for MCP tests
- Playwright browser installation
- OpenAI API key if using actual LLM calls

## Adding New Tests

To add new tests to the suite:

1. Create a new test function following the pattern:
```python
def test_new_feature():
    """Test description"""
    print("=" * 60)
    print("TEST: New Feature")
    print("=" * 60)
    
    try:
        # Test implementation
        print("✓ New feature test PASSED")
        return True
    except Exception as e:
        print(f"✗ New feature test FAILED: {e}")
        return False
```

2. Add the test to the `tests` list in `run_all_tests()`
3. Add command-line option in `main()` if needed

## Contributing

When contributing new features to GPTPlugins4All, please:
1. Add corresponding tests to this suite
2. Ensure all existing tests still pass
3. Update this README if adding new test categories
4. Follow the existing test patterns and naming conventions
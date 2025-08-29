#!/usr/bin/env python3
"""
MCP Integration Example for GPTPlugins4All

This example demonstrates how to set up and use MCP (Model Context Protocol)
integration with GPTPlugins4All to create powerful AI assistants with
web scraping and browser automation capabilities.
"""

import sys
import os
sys.path.append('..')
sys.path.append('../GPTPlugins4All')

from assistant import Assistant

def basic_mcp_example():
    """Basic example of MCP integration with Playwright and Firecrawl"""
    print("🚀 Basic MCP Integration Example")
    print("=" * 50)
    
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
                "FIRECRAWL_API_URL": "https://api.firecrawl.dev"  # Optional: custom instance
            }
        }
    }
    
    # Create assistant with MCP capabilities
    assistant = Assistant(
        configs=[],  # No API configs needed for this example
        name="Web Research Assistant",
        instructions="""You are a web research assistant with powerful capabilities.
        
        You can:
        - Navigate websites and take screenshots using Playwright
        - Scrape content from any website using Firecrawl
        - Search the web and extract relevant information
        - Fill out forms and interact with web pages
        - Crawl entire websites for comprehensive data
        
        Always choose the most appropriate tool for each task.""",
        model="gpt-4o-mini",  # Use a cost-effective model for examples
        old_mode=True,
        max_tokens=2000,
        mcp_servers=mcp_servers
    )
    
    print(f"✅ Assistant created with {len(assistant.mcp_tools)} MCP tools")
    
    # List available tools
    print("\n📋 Available MCP Tools:")
    for tool in assistant.mcp_tools[:10]:  # Show first 10 tools
        print(f"  - {tool['function']['name']}")
    
    if len(assistant.mcp_tools) > 10:
        print(f"  ... and {len(assistant.mcp_tools) - 10} more tools")
    
    return assistant

def advanced_mcp_example():
    """Advanced example with custom configuration and conditional loading"""
    print("\n🔧 Advanced MCP Configuration Example")
    print("=" * 50)
    
    # Advanced MCP configuration with environment-based loading
    mcp_servers = {}
    
    # Always include Playwright for browser automation
    mcp_servers["playwright"] = {
        "command": "npx",
        "args": ["@playwright/mcp@latest"]
    }
    
    # Conditionally include Firecrawl if API key is available
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if firecrawl_api_key:
        mcp_servers["firecrawl"] = {
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"],
            "env": {"FIRECRAWL_API_KEY": firecrawl_api_key}
        }
        print("✅ Firecrawl configured with API key")
    else:
        print("⚠️  Firecrawl API key not found, using default configuration")
        mcp_servers["firecrawl"] = {
            "command": "npx",
            "args": ["-y", "firecrawl-mcp"]
        }
    
    # Add filesystem access if directory is specified
    allowed_dir = os.getenv("MCP_FILESYSTEM_DIR", "/tmp")
    if os.path.exists(allowed_dir):
        mcp_servers["filesystem"] = {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", allowed_dir]
        }
        print(f"✅ Filesystem access configured for: {allowed_dir}")
    
    # Create assistant with advanced configuration
    assistant = Assistant(
        configs=[],
        name="Advanced MCP Assistant",
        instructions="""You are an advanced AI assistant with comprehensive web and file system capabilities.
        
        Your tools include:
        - Complete browser automation (navigate, click, type, screenshot)
        - Advanced web scraping and crawling
        - File system operations (if configured)
        - Web search with content extraction
        - Form filling and web interaction
        
        Use these tools intelligently to help users with research, data collection, and web automation tasks.""",
        model="gpt-4o",
        old_mode=True,
        max_tokens=3000,
        mcp_servers=mcp_servers
    )
    
    print(f"✅ Advanced assistant created with {len(assistant.mcp_tools)} MCP tools")
    
    # Categorize tools by server
    playwright_tools = [t for t in assistant.mcp_tools if 'playwright' in t['function']['name']]
    firecrawl_tools = [t for t in assistant.mcp_tools if 'firecrawl' in t['function']['name']]
    filesystem_tools = [t for t in assistant.mcp_tools if 'filesystem' in t['function']['name']]
    
    print(f"\n📊 Tool Distribution:")
    print(f"  🎭 Playwright: {len(playwright_tools)} tools")
    print(f"  🔥 Firecrawl: {len(firecrawl_tools)} tools")
    print(f"  📁 Filesystem: {len(filesystem_tools)} tools")
    
    return assistant

def demonstrate_tool_usage():
    """Demonstrate how MCP tools are automatically available"""
    print("\n🛠️  MCP Tool Usage Demonstration")
    print("=" * 50)
    
    print("MCP tools are automatically integrated into the assistant's capabilities.")
    print("No additional configuration is needed - they work like regular tools!")
    print("")
    print("Example conversation flow:")
    print("User: 'Please scrape the latest news from example.com'")
    print("Assistant: Uses mcp_firecrawl_scrape automatically")
    print("")
    print("User: 'Take a screenshot of that website'")
    print("Assistant: Uses mcp_playwright_browser_screenshot automatically")
    print("")
    print("User: 'Fill out the contact form with my details'")
    print("Assistant: Uses mcp_playwright_browser_fill_form automatically")
    print("")
    print("✨ The AI chooses the right tool based on the task!")

def main():
    """Run all MCP examples"""
    print("🧪 GPTPlugins4All MCP Integration Examples")
    print("=" * 60)
    
    try:
        # Basic example
        basic_assistant = basic_mcp_example()
        
        # Advanced example
        advanced_assistant = advanced_mcp_example()
        
        # Usage demonstration
        demonstrate_tool_usage()
        
        print("\n🎉 All examples completed successfully!")
        print("\nNext steps:")
        print("1. Try interacting with the assistants")
        print("2. Experiment with different MCP server configurations")
        print("3. Build your own MCP-enabled applications")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MCP is installed: pip install mcp")
        print("2. Install MCP servers: npm install -g @playwright/mcp firecrawl-mcp")
        print("3. Run the setup script: ./setup_mcp.sh")
        print("4. Check the README.md for detailed setup instructions")

if __name__ == "__main__":
    main()
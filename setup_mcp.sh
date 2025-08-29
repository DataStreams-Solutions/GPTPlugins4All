#!/bin/bash

# GPTPlugins4All MCP Setup Script
# This script installs the necessary dependencies for MCP integration

echo "🚀 GPTPlugins4All MCP Setup"
echo "=========================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python not found. Please install Python first."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js not found. Please install Node.js first."
    echo "   Visit: https://nodejs.org/ or use a package manager:"
    echo "   - macOS: brew install node"
    echo "   - Ubuntu: sudo apt install nodejs npm"
    echo "   - Windows: choco install nodejs"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm not found. Please install npm."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Install Python MCP library
echo "📦 Installing Python MCP library..."
pip install mcp
if [ $? -eq 0 ]; then
    echo "✅ MCP library installed successfully"
else
    echo "❌ Failed to install MCP library"
    exit 1
fi

echo ""

# Install popular MCP servers
echo "🔧 Installing popular MCP servers..."

echo "  Installing Playwright MCP (browser automation)..."
npm install -g @playwright/mcp
if [ $? -eq 0 ]; then
    echo "  ✅ Playwright MCP installed"
else
    echo "  ⚠️  Playwright MCP installation failed"
fi

echo "  Installing Firecrawl MCP (web scraping)..."
npm install -g firecrawl-mcp
if [ $? -eq 0 ]; then
    echo "  ✅ Firecrawl MCP installed"
else
    echo "  ⚠️  Firecrawl MCP installation failed"
fi

echo "  Installing Filesystem MCP..."
npm install -g @modelcontextprotocol/server-filesystem
if [ $? -eq 0 ]; then
    echo "  ✅ Filesystem MCP installed"
else
    echo "  ⚠️  Filesystem MCP installation failed"
fi

echo ""

# Install Playwright browsers if Playwright MCP was installed
if command -v playwright &> /dev/null; then
    echo "🌐 Installing Playwright browsers..."
    playwright install chromium
    if [ $? -eq 0 ]; then
        echo "✅ Playwright browsers installed"
    else
        echo "⚠️  Playwright browsers installation failed"
    fi
else
    echo "⚠️  Playwright not found, skipping browser installation"
fi

echo ""
echo "🎉 MCP setup completed!"
echo ""
echo "Next steps:"
echo "1. Test your setup: python test_suite.py mcp"
echo "2. Check the README.md for usage examples"
echo "3. Start building with MCP-enabled assistants!"
echo ""
echo "Example usage:"
echo "```python"
echo "from GPTPlugins4All.assistant import Assistant"
echo ""
echo "mcp_servers = {"
echo "    'playwright': {'command': 'npx', 'args': ['@playwright/mcp@latest']},"
echo "    'firecrawl': {'command': 'npx', 'args': ['-y', 'firecrawl-mcp']}"
echo "}"
echo ""
echo "assistant = Assistant("
echo "    configs=[],"
echo "    name='MCP Assistant',"
echo "    instructions='You have web scraping and browser automation tools.',"
echo "    model='gpt-4o',"
echo "    old_mode=True,"
echo "    mcp_servers=mcp_servers"
echo ")"
echo "```"
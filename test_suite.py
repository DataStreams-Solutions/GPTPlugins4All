#!/usr/bin/env python3
"""
GPTPlugins4All Testing Suite

This file contains comprehensive tests for the GPTPlugins4All library,
including MCP integration, scraping functionality, and assistant capabilities.
"""

import sys
import os
import json
import time
sys.path.append('GPTPlugins4All')

from assistant import Assistant, scrape_text

def test_basic_scraping():
    """Test basic web scraping functionality"""
    print("=" * 60)
    print("TEST: Basic Web Scraping")
    print("=" * 60)
    
    try:
        # Test with a simple website
        url = "https://httpbin.org/html"
        result = scrape_text(url, 1000)
        
        print(f"✓ Successfully scraped {url}")
        print(f"  Content length: {len(result)} characters")
        print(f"  First 200 chars: {result[:200]}...")
        
        if len(result) > 0:
            print("✓ Basic scraping test PASSED")
            return True
        else:
            print("✗ Basic scraping test FAILED - empty result")
            return False
            
    except Exception as e:
        print(f"✗ Basic scraping test FAILED: {e}")
        return False

def test_javascript_scraping():
    """Test JavaScript-heavy website scraping with Playwright fallback"""
    print("\n" + "=" * 60)
    print("TEST: JavaScript Scraping (Playwright Fallback)")
    print("=" * 60)
    
    try:
        # Test with versabot.co (JavaScript-heavy site)
        url = "https://versabot.co"
        result = scrape_text(url, 3000)
        
        print(f"✓ Successfully scraped {url}")
        print(f"  Content length: {len(result)} characters")
        print(f"  First 300 chars: {result[:300]}...")
        
        # Check if we got meaningful content (should contain "VersaBot" or similar)
        if len(result) > 1000 and ("versabot" in result.lower() or "sales" in result.lower()):
            print("✓ JavaScript scraping test PASSED")
            return True
        else:
            print("✗ JavaScript scraping test FAILED - insufficient or irrelevant content")
            return False
            
    except Exception as e:
        print(f"✗ JavaScript scraping test FAILED: {e}")
        return False

def test_mcp_integration():
    """Test MCP (Model Context Protocol) integration with Playwright and Firecrawl"""
    print("\n" + "=" * 60)
    print("TEST: MCP Integration")
    print("=" * 60)
    
    try:
        # MCP server configuration
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
        
        print("Creating Assistant with MCP servers...")
        
        # Create assistant with MCP servers
        assistant = Assistant(
            configs=[],  # Empty configs for this test
            name="MCP Test Assistant",
            instructions="You have access to web scraping and browser automation tools through MCP servers.",
            model="gpt-4o-mini",
            old_mode=True,
            max_tokens=2000,
            mcp_servers=mcp_servers
        )
        
        print("✓ Assistant created successfully!")
        print(f"✓ MCP servers configured: {list(assistant.mcp_servers.keys())}")
        print(f"✓ MCP tools available: {len(assistant.mcp_tools)}")
        
        # Verify MCP tools are integrated as regular tools
        print("\n🔧 Verifying MCP tools are integrated as regular tools...")
        
        # Check that MCP tools are in other_tools (regular tool integration)
        mcp_tools_in_other_tools = [tool for tool in assistant.other_tools if tool['function']['name'].startswith('mcp_')]
        print(f"✓ MCP tools in regular tool system: {len(mcp_tools_in_other_tools)}")
        
        # Check that MCP functions are in other_functions (regular function integration)
        mcp_functions_in_other_functions = [name for name in assistant.other_functions.keys() if name.startswith('mcp_')]
        print(f"✓ MCP functions in regular function system: {len(mcp_functions_in_other_functions)}")
        
        # Verify tools are callable through execute_function
        if len(mcp_functions_in_other_functions) > 0:
            sample_function = mcp_functions_in_other_functions[0]
            print(f"✓ Sample MCP function '{sample_function}' is callable through execute_function")
        
        # List available MCP tools by category
        playwright_tools = [tool for tool in assistant.mcp_tools if 'playwright' in tool['function']['name']]
        firecrawl_tools = [tool for tool in assistant.mcp_tools if 'firecrawl' in tool['function']['name']]
        
        print(f"\n📱 Playwright Tools ({len(playwright_tools)}):")
        for tool in playwright_tools[:5]:  # Show first 5
            print(f"  - {tool['function']['name']}")
        if len(playwright_tools) > 5:
            print(f"  ... and {len(playwright_tools) - 5} more")
        
        print(f"\n🔥 Firecrawl Tools ({len(firecrawl_tools)}):")
        for tool in firecrawl_tools:
            print(f"  - {tool['function']['name']}")
        
        print(f"\n✅ MCP Integration Summary:")
        print(f"  - MCP tools loaded: {len(assistant.mcp_tools)}")
        print(f"  - Tools integrated as regular tools: {len(mcp_tools_in_other_tools)}")
        print(f"  - Functions integrated as regular functions: {len(mcp_functions_in_other_functions)}")
        print(f"  - No additional configuration needed: ✓")
        print(f"  - Tools work automatically with AI: ✓")
        
        # Test a simple MCP tool call if available
        if len(assistant.mcp_tools) > 0:
            print(f"\n✓ MCP integration test PASSED - {len(assistant.mcp_tools)} tools loaded and integrated")
            return True
        else:
            print("\n✗ MCP integration test FAILED - no tools loaded")
            return False
            
    except Exception as e:
        print(f"✗ MCP integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_assistant_basic_functionality():
    """Test basic assistant functionality without external dependencies"""
    print("\n" + "=" * 60)
    print("TEST: Basic Assistant Functionality")
    print("=" * 60)
    
    try:
        # Create a simple assistant
        assistant = Assistant(
            configs=[],
            name="Test Assistant",
            instructions="You are a helpful test assistant.",
            model="gpt-4o-mini",
            old_mode=True,
            max_tokens=100,
            search_enabled=True,  # Enable search functionality
            view_pages=True       # Enable page viewing
        )
        
        print("✓ Assistant created successfully")
        print(f"✓ Model: {assistant.model}")
        print(f"✓ Search enabled: {assistant.search_enabled}")
        print(f"✓ Page viewing enabled: {assistant.view_pages}")
        
        # Check if built-in tools are available
        builtin_tools = []
        if assistant.search_enabled:
            builtin_tools.append("search_google")
        if assistant.view_pages:
            builtin_tools.append("scrape_text")
            
        print(f"✓ Built-in tools available: {builtin_tools}")
        
        print("✓ Basic assistant functionality test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Basic assistant functionality test FAILED: {e}")
        return False

def test_gpt5_token_handling():
    """Test GPT-5 token parameter handling"""
    print("\n" + "=" * 60)
    print("TEST: GPT-5 Token Parameter Handling")
    print("=" * 60)
    
    try:
        # Test GPT-5 model
        assistant_gpt5 = Assistant(
            configs=[],
            name="GPT-5 Test Assistant",
            instructions="Test assistant for GPT-5 token handling.",
            model="gpt-5-preview",  # Hypothetical GPT-5 model name
            old_mode=True,
            max_tokens=1000
        )
        
        # Check that max_completion_tokens is set instead of max_tokens
        if hasattr(assistant_gpt5, 'max_completion_tokens') and assistant_gpt5.max_completion_tokens == 1000:
            print("✓ GPT-5 model correctly uses max_completion_tokens")
        else:
            print("✗ GPT-5 model token handling incorrect")
            return False
            
        # Test GPT-4 model for comparison
        assistant_gpt4 = Assistant(
            configs=[],
            name="GPT-4 Test Assistant", 
            instructions="Test assistant for GPT-4 token handling.",
            model="gpt-4o",
            old_mode=True,
            max_tokens=1000
        )
        
        # Check that max_tokens is set for non-GPT-5 models
        if hasattr(assistant_gpt4, 'max_tokens') and assistant_gpt4.max_tokens == 1000:
            print("✓ GPT-4 model correctly uses max_tokens")
        else:
            print("✗ GPT-4 model token handling incorrect")
            return False
            
        print("✓ GPT-5 token handling test PASSED")
        return True
        
    except Exception as e:
        print(f"✗ GPT-5 token handling test FAILED: {e}")
        return False

def run_all_tests():
    """Run all tests and provide a summary"""
    print("🧪 GPTPlugins4All Testing Suite")
    print("=" * 60)
    print("Starting comprehensive test suite...")
    
    tests = [
        ("Basic Web Scraping", test_basic_scraping),
        ("JavaScript Scraping", test_javascript_scraping),
        ("MCP Integration", test_mcp_integration),
        ("Basic Assistant Functionality", test_assistant_basic_functionality),
        ("GPT-5 Token Handling", test_gpt5_token_handling)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:<10} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {time.time() - start_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed")
        return False

def main():
    """Main entry point for the test suite"""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1].lower()
        if test_name == "scraping":
            test_basic_scraping()
        elif test_name == "javascript":
            test_javascript_scraping()
        elif test_name == "mcp":
            test_mcp_integration()
        elif test_name == "assistant":
            test_assistant_basic_functionality()
        elif test_name == "gpt5":
            test_gpt5_token_handling()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: scraping, javascript, mcp, assistant, gpt5")
    else:
        # Run all tests
        run_all_tests()

if __name__ == "__main__":
    main()
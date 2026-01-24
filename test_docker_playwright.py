#!/usr/bin/env python3
"""
Test script to verify Playwright works in Docker environment
"""
import sys
import os
import pytest

if os.getenv("INTEGRATION") != "1":
    pytest.skip("Integration test (playwright); set INTEGRATION=1 to run", allow_module_level=True)
sys.path.append('GPTPlugins4All')

from assistant import scrape_text

def test_playwright_docker():
    """Test Playwright functionality in Docker"""
    print("🔧 Testing Playwright in Docker Environment")
    print("=" * 50)
    
    # Test URLs that require JavaScript rendering
    test_urls = [
        "https://versabot.co",
        "https://example.com",
        "https://httpbin.org/html"
    ]
    
    for url in test_urls:
        print(f"\n🌐 Testing: {url}")
        try:
            result = scrape_text(url, 3000)
            print(f"  ✅ Success: {len(result)} characters")
            if len(result) > 100:
                print(f"  📄 Preview: {result[:200]}...")
            else:
                print(f"  ⚠️  Short content: {result}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("Docker Playwright test completed")

if __name__ == "__main__":
    test_playwright_docker()

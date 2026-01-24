#!/usr/bin/env python3
"""
Specific test for versabot.co scraping issue

This test focuses on the versabot.co scraping problem to identify
why it returns empty strings in some environments.
"""

import sys
import os
import pytest
if os.getenv("INTEGRATION") != "1":
    pytest.skip("Integration test (versabot scraping); set INTEGRATION=1 to run", allow_module_level=True)
sys.path.append('GPTPlugins4All')

from assistant import scrape_text
import requests
import re

def test_versabot_http_request():
    """Test the initial HTTP request to versabot.co"""
    print("🌐 Testing HTTP Request to versabot.co")
    print("=" * 50)
    
    urls_to_test = [
        "https://versabot.co",
        "https://www.versabot.co"
    ]
    
    for url in urls_to_test:
        print(f"\nTesting {url}:")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
            response = requests.get(url, verify=False, headers=headers, allow_redirects=True)
            
            print(f"  Status: {response.status_code}")
            print(f"  Final URL: {response.url}")
            print(f"  Content length: {len(response.text)}")
            
            # Check for indicators
            has_links = bool(re.search(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', response.text))
            is_loading_page = bool(re.search(r'(loader|loading|spinner)', response.text.lower()))
            is_nextjs_app = bool(re.search(r'(__next|next\.js|_next/static)', response.text.lower()))
            has_minimal_content = len(re.sub(r'<[^>]+>', '', response.text).strip()) < 200
            
            print(f"  Has links: {has_links}")
            print(f"  Is loading page: {is_loading_page}")
            print(f"  Is Next.js app: {is_nextjs_app}")
            print(f"  Has minimal content: {has_minimal_content}")
            
            # Show first 500 characters
            print(f"  First 500 chars: {response.text[:500]}...")
            
        except Exception as e:
            print(f"  Error: {e}")

def test_versabot_scraping():
    """Test the scrape_text function with versabot.co"""
    print("\n🔧 Testing scrape_text Function")
    print("=" * 50)
    
    urls_to_test = [
        "https://versabot.co",
        "https://www.versabot.co"
    ]
    
    for url in urls_to_test:
        print(f"\nTesting scrape_text with {url}:")
        try:
            result = scrape_text(url, 3000)
            print(f"  Result length: {len(result)}")
            
            if len(result) > 0:
                print(f"  ✅ SUCCESS - Got content!")
                print(f"  First 300 chars: {result[:300]}...")
                
                # Check if it contains meaningful content
                if any(word in result.lower() for word in ['versabot', 'sales', 'automation', 'ai']):
                    print(f"  ✅ Contains relevant content")
                else:
                    print(f"  ⚠️  Content may not be fully loaded")
            else:
                print(f"  ❌ FAILED - Empty result!")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

def test_comparison_site():
    """Test with a comparison site that should work"""
    print("\n🆚 Testing Comparison Site (httpbin.org)")
    print("=" * 50)
    
    url = "https://httpbin.org/html"
    print(f"Testing scrape_text with {url}:")
    
    try:
        result = scrape_text(url, 3000)
        print(f"  Result length: {len(result)}")
        
        if len(result) > 0:
            print(f"  ✅ SUCCESS - Comparison site works")
            print(f"  First 200 chars: {result[:200]}...")
        else:
            print(f"  ❌ FAILED - Even comparison site failed!")
            
    except Exception as e:
        print(f"  ❌ ERROR: {e}")

def main():
    """Run all versabot-specific tests"""
    print("🎯 VersaBot.co Scraping Debug Suite")
    print("=" * 60)
    
    test_versabot_http_request()
    test_versabot_scraping()
    test_comparison_site()
    
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("If versabot.co returns empty but comparison site works:")
    print("1. The issue is site-specific (Next.js loading)")
    print("2. Playwright fallback should be triggered")
    print("3. Check if Playwright is waiting long enough")
    print("4. May need longer timeouts for JS rendering")
    print("")
    print("If both sites fail:")
    print("1. General Playwright/system issue")
    print("2. Check Docker environment setup")
    print("3. Verify Playwright installation")

if __name__ == "__main__":
    main()

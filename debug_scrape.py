#!/usr/bin/env python3
"""
Debug script for Playwright scraping issues in Docker

This script helps diagnose why Playwright scraping works locally but not in Docker.
Run this in your Docker container to identify the specific issue.
"""

import sys
import os
import json
import requests
from bs4 import BeautifulSoup

def debug_environment():
    """Check the environment and Playwright availability"""
    print("🔍 Environment Debug Information")
    print("=" * 50)
    
    # Check if running in Docker
    is_docker = os.path.exists('/.dockerenv')
    print(f"Running in Docker: {is_docker}")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if Playwright is installed
    try:
        from playwright.sync_api import sync_playwright
        print("✅ Playwright import: SUCCESS")
    except ImportError as e:
        print(f"❌ Playwright import: FAILED - {e}")
        return False
    
    # Check browser installation
    try:
        with sync_playwright() as p:
            browser_path = p.chromium.executable_path
            print(f"✅ Chromium path: {browser_path}")
            print(f"✅ Chromium exists: {os.path.exists(browser_path) if browser_path else 'Unknown'}")
    except Exception as e:
        print(f"❌ Browser check: FAILED - {e}")
        return False
    
    return True

def test_basic_browser_launch():
    """Test basic browser launch"""
    print("\n🚀 Basic Browser Launch Test")
    print("=" * 50)
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            print("Launching browser...")
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',
                    '--disable-gpu'
                ]
            )
            print("✅ Browser launched successfully")
            
            page = browser.new_page()
            print("✅ Page created successfully")
            
            browser.close()
            print("✅ Browser closed successfully")
            return True
            
    except Exception as e:
        print(f"❌ Browser launch failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_page_navigation():
    """Test page navigation and content extraction"""
    print("\n🌐 Page Navigation Test")
    print("=" * 50)
    
    test_url = "https://example.com"
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',
                    '--disable-gpu'
                ]
            )
            
            page = browser.new_page()
            
            # Set user agent
            page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            print(f"Navigating to {test_url}...")
            page.goto(test_url, timeout=60000, wait_until='networkidle')
            print("✅ Navigation successful")
            
            page.wait_for_selector("body", timeout=30000)
            print("✅ Body element found")
            
            content = page.content()
            print(f"✅ Content extracted: {len(content)} characters")
            
            if len(content) > 0:
                print(f"First 200 characters: {content[:200]}...")
            else:
                print("❌ Content is empty!")
            
            browser.close()
            return len(content) > 0
            
    except Exception as e:
        print(f"❌ Page navigation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_subprocess_approach():
    """Test the subprocess approach used in the library"""
    print("\n🔧 Subprocess Approach Test")
    print("=" * 50)
    
    import subprocess
    
    script = '''
import sys
import json
try:
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--no-first-run',
                '--no-zygote',
                '--single-process',
                '--disable-gpu'
            ]
        )
        page = browser.new_page()
        
        page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        page.goto("https://example.com", timeout=60000, wait_until='networkidle')
        page.wait_for_selector("body", timeout=30000)
        page.wait_for_timeout(2000)
        
        content = page.content()
        browser.close()
        
        print(json.dumps({"success": True, "content": content, "length": len(content)}))
        
except Exception as e:
    import traceback
    print(json.dumps({
        "success": False, 
        "error": str(e),
        "traceback": traceback.format_exc()
    }))
'''
    
    try:
        env = os.environ.copy()
        env.update({
            'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright',
            'DISPLAY': ':99'
        })
        
        result = subprocess.run(
            [sys.executable, '-c', script], 
            capture_output=True, 
            text=True, 
            timeout=120,
            env=env
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.strip())
                if data.get("success"):
                    print(f"✅ Subprocess test successful: {data.get('length', 0)} characters")
                    return True
                else:
                    print(f"❌ Subprocess test failed: {data.get('error')}")
                    if data.get('traceback'):
                        print(f"Traceback: {data.get('traceback')}")
                    return False
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse subprocess output: {e}")
                return False
        else:
            print(f"❌ Subprocess failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Subprocess test failed: {e}")
        return False

def test_versabot_scraping():
    """Test scraping versabot.co specifically"""
    print("\n🎯 VersaBot.co Scraping Test")
    print("=" * 50)
    
    url = "https://versabot.co"
    
    # First try regular requests
    print("Testing regular HTTP request...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
        response = requests.get(url, verify=False, headers=headers)
        print(f"HTTP response: {response.status_code}, {len(response.text)} characters")
        
        # Check for links
        import re
        has_links = bool(re.search(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', response.text))
        print(f"Has links: {has_links}")
        
        if not has_links:
            print("No links found - Playwright fallback would be triggered")
        
    except Exception as e:
        print(f"HTTP request failed: {e}")
    
    # Now try Playwright
    print("\nTesting Playwright approach...")
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--single-process',
                    '--disable-gpu'
                ]
            )
            
            page = browser.new_page()
            page.set_extra_http_headers({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            page.goto(url, timeout=60000, wait_until='networkidle')
            page.wait_for_selector("body", timeout=30000)
            page.wait_for_timeout(3000)  # Wait for dynamic content
            
            content = page.content()
            browser.close()
            
            print(f"✅ Playwright content: {len(content)} characters")
            
            if len(content) > 1000:
                # Extract text like the library does
                soup = BeautifulSoup(content, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = '\n'.join(chunk for chunk in chunks if chunk)
                
                print(f"✅ Extracted text: {len(text)} characters")
                print(f"First 300 characters: {text[:300]}...")
                
                return len(text) > 0
            else:
                print("❌ Content too short")
                return False
                
    except Exception as e:
        print(f"❌ Playwright test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests"""
    print("🐛 Playwright Docker Debug Suite")
    print("=" * 60)
    
    tests = [
        ("Environment Check", debug_environment),
        ("Basic Browser Launch", test_basic_browser_launch),
        ("Page Navigation", test_page_navigation),
        ("Subprocess Approach", test_subprocess_approach),
        ("VersaBot Scraping", test_versabot_scraping)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n🔧 Troubleshooting Recommendations:")
        print("1. Ensure all system dependencies are installed")
        print("2. Run: playwright install --with-deps chromium")
        print("3. Check Docker container has enough memory")
        print("4. Verify no conflicting processes")
        print("5. Try the improved Dockerfile.playwright")

if __name__ == "__main__":
    main()
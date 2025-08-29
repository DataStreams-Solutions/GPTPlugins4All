#!/usr/bin/env python3
"""
Debug script for Playwright Docker issues
"""
import os
import sys
import subprocess
import json

def check_playwright_installation():
    """Check if Playwright is properly installed"""
    print("🔍 Checking Playwright Installation")
    print("-" * 40)
    
    try:
        from playwright.sync_api import sync_playwright
        print("✅ Playwright import: SUCCESS")
        
        with sync_playwright() as p:
            try:
                browser_path = p.chromium.executable_path
                print(f"✅ Chromium path: {browser_path}")
                
                # Check if the executable exists and is executable
                if os.path.exists(browser_path):
                    print(f"✅ Browser executable exists")
                    if os.access(browser_path, os.X_OK):
                        print(f"✅ Browser executable is executable")
                    else:
                        print(f"❌ Browser executable is not executable")
                        print(f"   Permissions: {oct(os.stat(browser_path).st_mode)[-3:]}")
                else:
                    print(f"❌ Browser executable does not exist")
                    
            except Exception as e:
                print(f"❌ Browser path error: {e}")
                
    except ImportError as e:
        print(f"❌ Playwright import failed: {e}")

def check_environment():
    """Check environment variables"""
    print("\n🌍 Checking Environment")
    print("-" * 40)
    
    env_vars = [
        'PLAYWRIGHT_BROWSERS_PATH',
        'DISPLAY',
        'PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {value}")

def check_browser_directory():
    """Check browser directory contents"""
    print("\n📁 Checking Browser Directory")
    print("-" * 40)
    
    browser_path = '/ms-playwright'
    if os.path.exists(browser_path):
        print(f"✅ Browser directory exists: {browser_path}")
        try:
            contents = os.listdir(browser_path)
            print(f"Contents: {contents}")
            
            # Check for chromium directory
            for item in contents:
                if 'chromium' in item.lower():
                    chromium_path = os.path.join(browser_path, item)
                    print(f"Found Chromium: {chromium_path}")
                    if os.path.isdir(chromium_path):
                        chromium_contents = os.listdir(chromium_path)
                        print(f"Chromium contents: {chromium_contents}")
        except Exception as e:
            print(f"❌ Error listing directory: {e}")
    else:
        print(f"❌ Browser directory does not exist: {browser_path}")

def test_browser_launch():
    """Test browser launch with various configurations"""
    print("\n🚀 Testing Browser Launch")
    print("-" * 40)
    
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
            
            page.goto('https://example.com', timeout=30000)
            print("✅ Navigation successful")
            
            content = page.content()
            print(f"✅ Content retrieved: {len(content)} characters")
            
            browser.close()
            print("✅ Browser closed successfully")
            
    except Exception as e:
        print(f"❌ Browser launch failed: {e}")
        import traceback
        traceback.print_exc()

def test_subprocess_approach():
    """Test the subprocess approach used in the scraper"""
    print("\n🔧 Testing Subprocess Approach")
    print("-" * 40)
    
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
        page.goto("https://example.com", timeout=30000)
        content = page.content()
        browser.close()
        
        print(json.dumps({"success": True, "length": len(content)}))
        
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
            'DISPLAY': ':99',
            'PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD': '0'
        })
        
        result = subprocess.run(
            [sys.executable, '-c', script], 
            capture_output=True, 
            text=True, 
            timeout=60,
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
                else:
                    print(f"❌ Subprocess test failed: {data.get('error')}")
                    if data.get('traceback'):
                        print(f"Traceback: {data.get('traceback')}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
        else:
            print(f"❌ Subprocess failed with return code {result.returncode}")
            
    except Exception as e:
        print(f"❌ Subprocess error: {e}")

if __name__ == "__main__":
    print("🐳 Docker Playwright Debug Tool")
    print("=" * 50)
    
    check_playwright_installation()
    check_environment()
    check_browser_directory()
    test_browser_launch()
    test_subprocess_approach()
    
    print("\n" + "=" * 50)
    print("Debug completed")
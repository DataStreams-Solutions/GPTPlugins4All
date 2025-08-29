#!/usr/bin/env python3
"""
Fix script for Playwright browser installation in Docker
"""
import subprocess
import sys
import os

def install_playwright_browsers():
    """Install Playwright browsers if missing"""
    print("🔧 Installing Playwright browsers...")
    
    try:
        # Install browsers
        result = subprocess.run([
            sys.executable, '-m', 'playwright', 'install', '--with-deps', 'chromium'
        ], check=True, capture_output=True, text=True, timeout=300)
        
        print("✅ Playwright browsers installed successfully")
        print(f"Output: {result.stdout}")
        
        # Set permissions
        browser_path = '/ms-playwright'
        if os.path.exists(browser_path):
            os.system(f'chmod -R 755 {browser_path}')
            print("✅ Browser permissions set")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Browser installation failed: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_installation():
    """Test if the installation worked"""
    print("🧪 Testing installation...")
    
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox']
            )
            page = browser.new_page()
            page.goto('https://example.com')
            content = page.content()
            browser.close()
            
            print(f"✅ Test successful: {len(content)} characters")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🐳 Playwright Docker Fix Tool")
    print("=" * 40)
    
    if install_playwright_browsers():
        if test_installation():
            print("✅ Playwright is now working correctly!")
        else:
            print("❌ Installation completed but test failed")
    else:
        print("❌ Installation failed")
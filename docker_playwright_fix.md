# Playwright Docker Fix for GPTPlugins4All

## Problem
Playwright scraping works locally but returns empty strings in Docker containers due to missing system dependencies and browser installation issues.

## Root Causes
1. Missing system dependencies for Chromium
2. Playwright browsers not properly installed in container
3. Subprocess execution issues in containerized environment
4. Missing display/graphics libraries

## Solution 1: Updated Dockerfile (Recommended)

Replace your current Dockerfile with this improved version:

```dockerfile
FROM python:3.10

WORKDIR /app

# Install system dependencies for Playwright
RUN apt-get update && apt-get install -y \
    swig \
    libpulse-dev \
    # Playwright/Chromium dependencies
    libnss3 \
    libnspr4 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libxss1 \
    libasound2 \
    libatspi2.0-0 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
COPY ./wheels /app/wheels

RUN pip install --no-cache-dir textract
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers with proper dependencies
RUN playwright install --with-deps chromium

# Verify Playwright installation
RUN python -c "from playwright.sync_api import sync_playwright; print('Playwright installed successfully')"

COPY . .

CMD ["gunicorn", "-k", "customworker.CustomGeventWebSocketWorker", "-w", "4", "app:app", "--preload", "-b", "0.0.0.0:8000", "--worker-connections", "1000", "--no-sendfile"]

EXPOSE 8000
```

## Solution 2: Enhanced Playwright Function

Update the `_playwright_scrape_subprocess` function to handle Docker environment better:

```python
def _playwright_scrape_subprocess(url):
    """Run Playwright in a subprocess with Docker-compatible settings"""
    import subprocess
    import sys
    import json
    import os
    
    # Create a more robust Python script for Docker environments
    script = f'''
import sys
import json
import os
try:
    from playwright.sync_api import sync_playwright
    
    # Configure for headless Docker environment
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
        
        # Set user agent to avoid bot detection
        page.set_extra_http_headers({{
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }})
        
        page.goto("{url}", timeout=60000, wait_until='networkidle')
        page.wait_for_selector("body", timeout=30000)
        
        # Wait a bit more for dynamic content
        page.wait_for_timeout(2000)
        
        content = page.content()
        browser.close()
        
        print(json.dumps({{"success": True, "content": content, "length": len(content)}}))
        
except Exception as e:
    import traceback
    error_details = {{
        "error": str(e),
        "traceback": traceback.format_exc(),
        "playwright_available": False
    }}
    
    try:
        from playwright.sync_api import sync_playwright
        error_details["playwright_available"] = True
    except ImportError:
        pass
        
    print(json.dumps({{"success": False, **error_details}}))
'''
    
    try:
        # Set environment variables for subprocess
        env = os.environ.copy()
        env.update({
            'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright',
            'DISPLAY': ':99'  # Virtual display for Docker
        })
        
        result = subprocess.run(
            [sys.executable, '-c', script], 
            capture_output=True, 
            text=True, 
            timeout=120,
            env=env
        )
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout.strip())
                if data.get("success"):
                    content = data.get("content", "")
                    print(f"Playwright subprocess success: {data.get('length', 0)} characters")
                    return content
                else:
                    print(f"Playwright subprocess error: {data.get('error')}")
                    print(f"Playwright available: {data.get('playwright_available')}")
                    if data.get('traceback'):
                        print(f"Traceback: {data.get('traceback')}")
                    return None
            except json.JSONDecodeError as e:
                print(f"Failed to parse subprocess output: {e}")
                print(f"Raw output: {result.stdout}")
                return None
        else:
            print(f"Subprocess failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Playwright subprocess timed out")
        return None
    except Exception as e:
        print(f"Error running Playwright subprocess: {e}")
        return None
```

## Solution 3: Alternative Dockerfile with Playwright Docker Image

Use the official Playwright Docker image as base:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.40.0-focal

WORKDIR /app

# Install additional dependencies
RUN apt-get update && apt-get install -y \
    swig \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
COPY ./wheels /app/wheels

RUN pip install --no-cache-dir textract
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-k", "customworker.CustomGeventWebSocketWorker", "-w", "4", "app:app", "--preload", "-b", "0.0.0.0:8000", "--worker-connections", "1000", "--no-sendfile"]

EXPOSE 8000
```

## Solution 4: Debug Version

Add debugging to see what's happening:

```python
def scrape_text_debug(url, length):
    """Debug version of scrape_text to diagnose Docker issues"""
    import re
    import os
    
    print(f"DEBUG: Scraping {url} in environment: {'Docker' if os.path.exists('/.dockerenv') else 'Local'}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    response = requests.get(url, verify=False, headers=headers)
    
    if not isinstance(length, int):
        length = 3000
    if response.status_code >= 400:
        return "Error: HTTP " + str(response.status_code) + " error"

    response_text = response.text
    has_links = bool(re.search(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"', response_text))
    content_too_small = len(response_text.strip()) < 100
    
    print(f"DEBUG: Initial response length: {len(response_text)}")
    print(f"DEBUG: Has links: {has_links}")
    print(f"DEBUG: Content too small: {content_too_small}")
    
    if content_too_small or not has_links:
        print(f"DEBUG: Triggering Playwright fallback")
        
        # Check if Playwright is available
        try:
            from playwright.sync_api import sync_playwright
            print("DEBUG: Playwright import successful")
        except ImportError as e:
            print(f"DEBUG: Playwright import failed: {e}")
            return response_text
        
        playwright_content = _playwright_scrape_subprocess(url)
        if playwright_content:
            response_text = playwright_content
            print(f"DEBUG: Playwright success, content length: {len(response_text)}")
        else:
            print("DEBUG: Playwright fallback failed, using original response")

    # Continue with normal processing...
    soup = BeautifulSoup(response_text, "html.parser")
    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    text = leftTruncate(text, length)
    
    print(f"DEBUG: Final text length: {len(text)}")
    return text
```

## Quick Test Commands

Add these to your container to test Playwright:

```bash
# Test Playwright installation
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"

# Test browser launch
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    page.goto('https://example.com')
    print('Browser test OK:', len(page.content()))
    browser.close()
"

# Check browser installation
playwright install --dry-run chromium
```

## Recommended Implementation

1. Use **Solution 1** (Updated Dockerfile) for the most reliable fix
2. Apply **Solution 2** (Enhanced function) for better error handling
3. Use **Solution 4** (Debug version) temporarily to diagnose the exact issue

The key changes are:
- Install system dependencies with `--with-deps`
- Add Chrome/Chromium specific arguments for Docker
- Better error handling and debugging
- Proper environment variables
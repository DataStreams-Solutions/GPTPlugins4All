# Playwright Docker Fix

## Problem
Playwright browsers are not being installed properly in Docker containers, causing the error:
```
Executable doesn't exist at /ms-playwright/chromium_headless_shell-1187/chrome-linux/headless_shell
```

## Root Cause
The issue occurs because:
1. Playwright browsers weren't being installed with the correct Python environment
2. Browser files didn't have proper permissions
3. The installation process wasn't being verified properly

## Solution

### 1. Updated Dockerfile.playwright
The key changes made:

```dockerfile
# Install Playwright first
RUN pip install playwright

# Create browser directory with proper permissions
RUN mkdir -p /ms-playwright && chmod 755 /ms-playwright

# Install Playwright browsers with proper dependencies
RUN python -m playwright install --with-deps chromium

# Set proper permissions for Playwright browsers
RUN chmod -R 755 /ms-playwright

# Verify browser installation
RUN ls -la /ms-playwright/

# Alternative browser installation method if the first one failed
RUN if [ ! -d "/ms-playwright/chromium_headless_shell-"* ]; then \
        echo "Retrying browser installation..."; \
        PLAYWRIGHT_BROWSERS_PATH=/ms-playwright python -m playwright install chromium; \
        chmod -R 755 /ms-playwright; \
    fi
```

### 2. Enhanced Subprocess Code
Updated the `_playwright_scrape_subprocess` function in `assistant.py` to:
- Check browser availability before launching
- Attempt browser installation if missing
- Use proper Docker-compatible browser arguments
- Better error handling and logging

### 3. Testing Tools
Created several debugging and testing tools:

- `debug_docker_playwright.py` - Comprehensive diagnostic tool
- `test_docker_playwright.py` - Simple test for the scraper
- `fix_playwright_docker.py` - Runtime fix tool
- `build_and_test_docker.sh` - Build and test script

## Usage

### Build and Test
```bash
chmod +x build_and_test_docker.sh
./build_and_test_docker.sh
```

### Debug Issues
```bash
docker run --rm playwright-scraper python debug_docker_playwright.py
```

### Runtime Fix (if needed)
```bash
docker run --rm playwright-scraper python fix_playwright_docker.py
```

## Key Changes Summary

1. **Explicit Playwright installation**: `pip install playwright` before browser installation
2. **Use python -m playwright install**: Ensures correct Python environment
3. **Proper permissions**: `chmod -R 755 /ms-playwright`
4. **Fallback installation**: Retry mechanism if first installation fails
5. **Enhanced error handling**: Better diagnostics and runtime browser installation
6. **Docker-compatible arguments**: Proper browser launch arguments for containers

## Environment Variables
The subprocess now sets these environment variables:
```python
env.update({
    'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright',
    'DISPLAY': ':99',
    'PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD': '0'
})
```

This should resolve the Playwright browser installation issues in Docker containers.
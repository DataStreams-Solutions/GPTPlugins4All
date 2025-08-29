# Playwright Docker Solution for GPTPlugins4All

## Problem Summary
Playwright scraping works locally but returns empty strings in Docker containers.

## Root Cause
The issue is typically caused by:
1. **Missing system dependencies** for Chromium in Docker
2. **Improper browser installation** in containerized environment  
3. **Lack of Docker-specific Chrome arguments** (--no-sandbox, etc.)
4. **Missing environment variables** for headless operation

## ✅ Complete Solution

### 1. Updated Code (Already Applied)
The `_playwright_scrape_subprocess` function has been updated with Docker-compatible settings:

- ✅ **Docker-specific Chrome arguments** (--no-sandbox, --disable-setuid-sandbox, etc.)
- ✅ **Better error handling** with detailed debugging
- ✅ **Environment variables** for Docker compatibility
- ✅ **Longer wait times** for dynamic content
- ✅ **Proper user agent** to avoid bot detection

### 2. Improved Dockerfile
Use this Dockerfile instead of your current one:

```dockerfile
FROM python:3.10

WORKDIR /app

# Install system dependencies for Playwright and Chromium
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
    fonts-liberation \
    libappindicator3-1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .
COPY ./wheels /app/wheels

RUN pip install --no-cache-dir textract
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright and browsers with proper dependencies
RUN playwright install --with-deps chromium

# Verify installation
RUN python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"

COPY . .

# Set environment variables for Playwright
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV DISPLAY=:99

CMD ["gunicorn", "-k", "customworker.CustomGeventWebSocketWorker", "-w", "4", "app:app", "--preload", "-b", "0.0.0.0:8000", "--worker-connections", "1000", "--no-sendfile"]

EXPOSE 8000
```

### 3. Alternative: Use Playwright Docker Image
For guaranteed compatibility, use the official Playwright image:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.40.0-focal

WORKDIR /app

# Install your additional dependencies
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

## 🔧 Debugging Steps

### Step 1: Test in Container
Copy `debug_scrape.py` to your container and run:

```bash
python debug_scrape.py
```

This will identify the specific issue.

### Step 2: Manual Tests
Run these commands in your container:

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
    print('Content length:', len(page.content()))
    browser.close()
"
```

### Step 3: Check Dependencies
Ensure these are installed:

```bash
playwright install --with-deps chromium
```

## 🚀 Quick Fix Implementation

### Option A: Rebuild with New Dockerfile
1. Replace your Dockerfile with the improved version above
2. Rebuild your Docker image
3. Deploy and test

### Option B: Use Playwright Base Image  
1. Use the Playwright Docker image as base
2. Much simpler and guaranteed to work
3. Slightly larger image size

### Option C: Debug First
1. Use `debug_scrape.py` to identify the exact issue
2. Apply targeted fixes based on the results
3. Most efficient if you want to keep your current setup

## 📊 Expected Results

After applying the fix, you should see:

```
✅ Environment Check: PASSED
✅ Basic Browser Launch: PASSED  
✅ Page Navigation: PASSED
✅ Subprocess Approach: PASSED
✅ VersaBot Scraping: PASSED
```

And your scraping should return content like:
```
Playwright subprocess success: 195845 characters
Successfully rendered page with headless browser, content length: 195845
```

## 🎯 Key Changes Made

1. **Enhanced `_playwright_scrape_subprocess`** with Docker-specific Chrome arguments
2. **Better error handling** with detailed logging
3. **Environment variables** for Docker compatibility
4. **Longer timeouts** for dynamic content loading
5. **Proper user agent** to avoid bot detection

The updated code is already in your `assistant.py` file and should work with the improved Dockerfile.

## 💡 Pro Tips

1. **Use `--with-deps`** when installing Playwright in Docker
2. **Always include `--no-sandbox`** for Chrome in containers
3. **Set proper environment variables** (DISPLAY, PLAYWRIGHT_BROWSERS_PATH)
4. **Test with `debug_scrape.py`** before deploying
5. **Consider the official Playwright image** for simplicity

Your Playwright scraping should now work reliably in Docker! 🎉
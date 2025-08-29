#!/bin/bash

echo "🐳 Building Docker image with Playwright support..."

# Build the Docker image
docker build -f Dockerfile.playwright -t playwright-scraper .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker build successful"

echo "🧪 Testing Playwright in Docker container..."

# Test the container
docker run --rm playwright-scraper python test_docker_playwright.py

if [ $? -eq 0 ]; then
    echo "✅ Docker Playwright test successful"
else
    echo "❌ Docker Playwright test failed"
    echo "🔍 Checking browser installation..."
    docker run --rm playwright-scraper ls -la /ms-playwright/
    echo "🔍 Checking Playwright installation..."
    docker run --rm playwright-scraper python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
fi
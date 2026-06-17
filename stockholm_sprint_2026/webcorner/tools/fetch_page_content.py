import random
import re
from playwright.sync_api import sync_playwright


def fetch_page_content(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        # 1. Grab only the visible text, stripping 100% of HTML tags
        raw_text = page.locator("body").inner_text()

        # 2. Clean up whitespace to compress the token count even further
        # This replaces multiple tabs/newlines/spaces with a single space or newline
        clean_text = re.sub(r'\n+', '\n', raw_text)  # Collapse multiple newlines
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Collapse multiple spaces/tabs
        clean_text = clean_text.strip()

        browser.close()
        logfileName = f"logfile_{str(random.randint(1001, 10000))}.log"
        with open(f"output/{logfileName}", "w+") as file:
            file.write(clean_text)

        return clean_text
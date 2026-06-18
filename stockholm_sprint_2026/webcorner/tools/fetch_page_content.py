import random
import re
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth  # Proper top-level class import


def fetch_page_content(url):
    clean_text = ""
    print(f"Function called: fetch_page_content for url: {url}")
    try:
        # 1. Wrap sync_playwright() inside the Stealth context manager
        # This guarantees all pages and contexts inherit stealth signatures automatically
        with Stealth().use_sync(sync_playwright()) as p:
            # Launch browser (Switch headless to False if you need to visually debug)
            browser = p.chromium.launch(headless=True)
            
            # 2. Set a realistic User-Agent so you don't broadcast "HeadlessChrome"
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 720}
            )
            
            page = context.new_page()
            
            # Step 3 (The manual stealth call) is handled automatically by the context manager above.
            
            # 4. Navigate with your existing domcontentloaded strategy
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # 5. Grab only the visible text, stripping 100% of HTML tags
            raw_text = page.locator("body").inner_text()
            
            # 6. Clean up whitespace to compress the token count even further
            clean_text = re.sub(r'\n+', '\n', raw_text)  # Collapse multiple newlines
            clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Collapse multiple spaces/tabs
            clean_text = clean_text.strip()
            
            browser.close()
            
    except Exception as e:
        print(f"Error or Timeout loading {url}: {e}")
        clean_text = ""
        
    # 7. Log output if content was successfully retrieved
    if clean_text:
        logfileName = f"logfile_{str(random.randint(1001, 10000))}.log"
        with open(f"output/{logfileName}", "w+", encoding="utf-8") as file:
            file.write(clean_text)

        print(f"Logged page content to {logfileName}")
            
    return clean_text
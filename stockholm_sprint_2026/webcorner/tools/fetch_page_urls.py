from playwright.sync_api import sync_playwright
from urllib.parse import urljoin, urlparse
import random
import os


def fetch_page_urls(url: str) -> list[str]:
    """
    Given a URL, uses Playwright to let JavaScript load, 
    and returns a list of all unique hyperlinks found in the final DOM.
    """
    print(f"Function called: fetch_page_urls for url: {url}")
    urls = set()
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Wait for the network to go idle so all JS-rendered links are loaded
            page.goto(url, wait_until="networkidle", timeout=15000)
            
            # Capture the CURRENT URL in case a redirect happened
            current_url = page.url
            # Query the live DOM directly for all 'a' tags with an 'href' attribute
            href_handles = page.locator('a[href]')
            href_list = [href_handles.nth(i).get_attribute('href') for i in range(href_handles.count())]
            
            browser.close()
            
    except Exception as e:
        print(f"Error fetching URL with Playwright: {str(e)}")
        return [f"Error fetching URL: {str(e)}"]

    # Process and clean the links exactly like your original script
    for href in href_list:
        if not href:
            continue
        full_url = urljoin(current_url, href)
        parsed = urlparse(full_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"

        if parsed.scheme in ('http', 'https'):
            urls.add(clean_url)
    
    if current_url != url:
        urls.add(current_url)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    logfileName = f"logfile_{str(random.randint(1, 1000))}.log"
    print(f"Logging file output to: output/{logfileName}")
    with open(f"output/{logfileName}", "w+", encoding="utf-8") as file:
        for url in sorted(list(urls)):
            file.write(f"{url}\n")

    return sorted(list(urls))

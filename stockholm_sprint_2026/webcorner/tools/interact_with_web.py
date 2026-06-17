import os
from playwright.sync_api import sync_playwright


def interact_with_web(url: str, action: str, selector: str, value: str = "") -> str:
    """
    Interacts with a webpage and records a video of the session.
    """
    print(f"Function called: interact_with_web(action={action}, selector={selector})")
    
    # Create a directory for videos if it doesn't exist
    video_dir = "recordings"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    with sync_playwright() as p:
        # Note: Even with headless=False, if you are using xvfb-run, 
        # you won't see the window, but the video will still record.
        browser = p.chromium.launch(headless=True)
        
        # --- THE KEY CHANGE IS HERE ---
        context = browser.new_context(
            record_video_dir=video_dir
        )
        # ------------------------------
        
        page = context.new_page()
        
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            
            if action == "click":
                page.click(selector, timeout=5000)
            elif action == "type":
                page.fill(selector, value, timeout=5000)
            elif action == "scroll":
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            page.wait_for_load_state("networkidle")
            
            content = page.evaluate("document.body.innerText")
            return f"URL: {page.url}\n\nContent:\n{content}"
            
        except Exception as e:
            return f"Interaction failed: {str(e)}"
        finally:
            # IMPORTANT: The video file is only written to disk when the context/browser closes
            context.close() 
            browser.close()
            print(f"Session finished. Check the '{video_dir}' folder for the video.")

# Example usage:
# result = interact_with_web("https://google.com", "type", "textarea[name='q']", "Playwright Python")
# print(result)
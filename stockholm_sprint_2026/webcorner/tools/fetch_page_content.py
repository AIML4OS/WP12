import requests
import random
from bs4 import BeautifulSoup


def fetch_page_content(url: str):
    print(f"Function called: fetch_page_content for url {url}")
    """
    Fetches the text content only from a given URL.
        """
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements from the text extraction
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Get text and clean up whitespace
        text = soup.get_text(separator=' ')
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        logfileName = f"logfile_{str(random.randint(1001, 10000))}.log"
        print(f"Logging file output (fetch content) to: {logfileName}")
        with open(f"output/{logfileName}", "w+") as file:
            file.write(text)
        return text
    except Exception as e:
        return f"Error fetching content: {e}"

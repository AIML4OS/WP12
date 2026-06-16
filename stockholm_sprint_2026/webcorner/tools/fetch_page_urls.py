import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


def fetch_page_urls(url: str) -> list[str]:
    """
    Given a URL, returns a list of all unique hyperlinks found on that page.
    """
    print(f"Function called: fetch_page_urls for url: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching URL: {str(e)}")
        return [f"Error fetching URL: {str(e)}"]

    soup = BeautifulSoup(response.text, 'html.parser')
    urls = set()

    for link in soup.find_all('a', href=True):
        href = link.get('href')
        # Join relative URLs with the base URL
        full_url = urljoin(url, href)

        # Clean the URL (remove fragments like #section)
        parsed = urlparse(full_url)
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"

        # Only add if it has a scheme (http/https)
        if parsed.scheme in ('http', 'https'):
            urls.add(clean_url)

    return sorted(list(urls))

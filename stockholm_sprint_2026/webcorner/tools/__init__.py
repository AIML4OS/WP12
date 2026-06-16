from .fetch_page_urls import fetch_page_urls
from .fetch_page_content import fetch_page_content


def get_tools():
    return {
        'fetch_page_urls': fetch_page_urls,
        'fetch_page_content': fetch_page_content
    }


def get_tool_dict():
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_page_urls",
                "description": "Fetches all unique hyperlinks from a given website URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to scan for links."
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_page_content",
                "description": "Fetches the text content only from a given URL, removing scripts and styles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to fetch text content from."
                        }
                    },
                    "required": ["url"]
                }
            }
        }
    ]
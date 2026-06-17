from .fetch_page_urls import fetch_page_urls
from .fetch_page_content import fetch_page_content
from .interact_with_web import interact_with_web


def get_tools():
    return {
        'fetch_page_urls': fetch_page_urls,
        'fetch_page_content': fetch_page_content,
        # 'interact_with_web': interact_with_web
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
                "description": "Fetches the text content from a given URL.",
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
        },
        # {
        #     "type": "function",
        #     "function": {
        #         "name": "interact_with_web",
        #         "description": "Interacts with a webpage (click, type, scroll) to handle dynamic content.",
        #         "parameters": {
        #             "type": "object",
        #             "properties": {
        #                 "url": {
        #                     "type": "string",
        #                     "description": "The URL of the webpage to interact with."
        #                 },
        #                 "action": {
        #                     "type": "string",
        #                     "enum": ["click", "type", "scroll"],
        #                     "description": "The action to perform."
        #                 },
        #                 "selector": {
        #                     "type": "string",
        #                     "description": "The CSS selector for the element to interact with."
        #                 },
        #                 "value": {
        #                     "type": "string",
        #                     "description": "The text to type (only required if action is 'type')."
        #                 }
        #             },
        #             "required": ["url", "action", "selector"]
        #         }
        #     }
        # }
    ]
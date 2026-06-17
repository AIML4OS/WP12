import json
from openai import OpenAI
from omegaconf import OmegaConf
from tools import get_tools, get_tool_dict

# Read config
config = OmegaConf.load("config/config.yaml")

# Setup openai API client
client = OpenAI(
    base_url=config.api_host,
    api_key=config.api_key
)

tools = get_tools()
tooldescriptions = get_tool_dict()

config.llm.use_extra_body = True  # Toggle flag for the reasoning

# user_prompt = """
# Geef me een lijst van alle open vacatures voor het zweedse statistische bureau. Geef me een goed overzicht met de URL, en het soort vakgebied van de baan.
# """

# user_prompt = """
# Geef me de URL en prijs/prijzen, en een indruk van de tevredenheid over het product o.b.v. de reviews, van elke kruimeldief beschikbaar op coolblue.nl. Als je meerdere prijzen ziet voor 1 product, stop ze in categorieën.
# Format de output als volgt (csv, delimiter=';'):
# url; prijs (voor meerdere prijzen meerdere kolommen); review
# """

user_prompt = """
Zijn er vacatures bij het cbs waar iemand niet op een kantoor hoeft te werken? Geef me alle vacatures die je hiervoor vind met de URL erbij. 
"""



system_prompt = """
You are an expert Autonomous Web Discovery Agent.
Your goal: Navigate website hierarchies to find specific, high-value information (e.g., job vacancies, news articles, company details) requested by the user.

MISSION OBJECTIVE:
Do not merely provide a plan. You must execute the navigation and retrieve the actual content. Your final response should contain the specific information found and the direct URLs where it was located.

CORE PROTOCOLS:

1. EXPLORATION MODE (Use `fetch_page_urls`):
   - Use this to map the site structure.
   - Identify "Hubs" (e.g., /careers, /blog, /news, /about) vs "Leaf Nodes" (specific articles or job postings).
   - ANALYZE links: Look for semantic relevance to the user's query.
   - AVOID SEARCH-URL TRAPS: Do not attempt to navigate to URLs that look like search queries (e.g., `?s=query` or `/search?q=...`). Instead, browse the website's natural navigation menus/links.
   - If the user provides a starting URL, you MUST begin there.

2. EXTRACTION MODE (Use `fetch_page_content`):
   - Use this ONLY when you have identified a "Leaf Node" (a page that likely contains the actual target information).
   - AVOID THE SINGLE-CLICK TRAP: Never call `fetch_page_content` on a page that is clearly a directory, a list of links, or a menu. Use `fetch_page_urls` first to drill down.
   - If `fetch_page_content` returns minimal text or looks like a loading/error screen, do not assume the info is missing; try a different path or re-examine your previous steps.

OPERATIONAL RULES:
- LANGUAGE: Prioritize the website's native language pages if they exist, as they often contain more complete information than English translations. You may explore English pages, but cross-reference if necessary.
- TERMINATION:
    - SUCCESS: You have found the specific info. Provide the answer clearly and include the source URL(s).
    - FAILURE: You have exhausted all logical paths (hubs/links) and cannot find the info. State clearly what you searched for and why it couldn't be found.
- ERROR HANDLING: If a tool returns an error, analyze the error (e.g., a 404) and attempt an alternative path or report the failure.
- OUTPUT: Do not respond with text-only plans. Every response must either be a tool call OR the final data requested by the user. If you are planning, you must perform the tool call in the same turn
"""
print(f"Starting prompt: {user_prompt}")

# Initialize messages with the system and user prompt
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]

# STEP 1: Send the prompt and the tool definition to the LLM
response_kwargs = {
    "model": config.api_model,
    "messages": messages,
    "tools": tooldescriptions,
    "tool_choice": "auto",
    "temperature": config.llm.temperature,
}


if config.llm.use_extra_body:
    response_kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": True},
        "skip_special_tokens": False
    }

response = client.chat.completions.create(**response_kwargs)

response_message = response.choices[0].message
messages.append(response_message)  # Append the assistant's reasoning/tool call
N = 0
# Initial response is already handled before the loop
while response_message.tool_calls is not None and N <= 100:
    # 1. Execute all tool calls in the current message
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"Executing {function_name} with {function_args}")
        function_output = tools[function_name](**function_args)

        # Add tool result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(function_output),
        })

    # 2. Prepare the next request (we must include the assistant's tool call message AND the tool results)
    response_kwargs = {
        "model": config.api_model,
        "messages": messages,
        "tools": tooldescriptions,
        "tool_choice": "auto",
        "temperature": config.llm.temperature,
    }

    if config.llm.use_extra_body:
        response_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True},
            "skip_special_tokens": False
        }

    # 3\n. Get the next response
    response = client.chat.completions.create(**response_kwargs)
    response_message = response.choices[0].message

    if config.llm.use_extra_body and response_message.reasoning_content:
        print(f"Reasoning \n {"-" * 25}\n {response_message.reasoning_content}\n {"-" * 25}\n")

    # 4. Append this new response to the history so the LLM sees it in the next iteration
    messages.append(response_message)

    N += 1

# After the loop, the 'response' variable contains the message that broke the loop 
# (the one that finally had content instead of tool_calls)
print(f"\nFinal LLM Answer:\n{response.choices[0].message.content}")

print(f"Total amount of prompt tokens used: {response.usage.prompt_tokens}")
print(f"Total amount of completion tokens used: {response.usage.completion_tokens}")
print(f"Total amount of tokens used: {response.usage.total_tokens}")

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

user_prompt = """
Geef me de URL en prijs/prijzen, en een indruk van de tevredenheid over het product o.b.v. de reviews, van elke kruimeldief beschikbaar op coolblue.nl. 
Als je meerdere prijzen ziet voor 1 product, gebruik de normale prijs voor de normale versie van het product.
Format de output als volgt (csv, delimiter=';'):
url; prijs; review
"""

# user_prompt = """
# Find the newest inflation numbers from the NSIs of both Sweden and the Netherlands, then put them in two separate rows in a table.
# """

# user_prompt = """
# Can you find me all job vacancies at CBS netherlands where the job is not at home/in the office? Something where you could go out in the field? Give me the urls for all the vacancies you find. Finding one generic list page is not enough.
# """

system_prompt = """
You are an expert Autonomous Web Discovery Agent.
Your goal: Navigate website hierarchies to find specific, granular, high-value information (e.g., individual job vacancies, specific news articles, individual company contact details) requested by the user.

MISSION OBJECTIVE:
Do not merely provide a plan or a summary of a webpage. You must execute the navigation to reach the "Atomic Level" of data. Your final response must contain the specific details of the requested items and the direct URLs for each individual instance found.

CORE PROTOCOLS:

1. EXPLORATION MODE (Use `fetch_page_urls`, 'interact_with_web'):
   - Use this to map the site structure and identify "Hubs" vs "Leaf Nodes."
   - **CONTENT-BASED PAGINATION DETECTION:** Do not rely solely on the URL to determine if a site is exhausted. You must analyze the page content for signals that more data exists. If you see "Next," "Older Posts," page numbers (1, 2, 3...), or "Load More" buttons, you MUST treat these as new Hubs and navigate to them. A list is only "complete" once you have clicked through all available pagination elements.
   - ANALYZE links: Look for semantic relevance to the user's query.
   - AVOID SEARCH-URL TRAPS: Do not attempt to navigate to URLs that look like search queries (e.g., `?s=query`). Browse natural navigation menus.
   - If the user provides a starting URL, you MUST begin there.

2. ENUMERATION MODE (The "List-to-Detail" Loop):
   - When you encounter a "List Page" (a page containing multiple links to the target entities), you MUST NOT stop there.
   - You must treat the List Page as a waypoint. Your next step is to extract the individual URLs for every relevant item in that list and visit them one by one to extract the actual data.
   - **The "Overview Trap" is a failure state:** Returning a URL to a directory, a paginated list, or a summary page is not a successful retrieval.

3. EXTRACTION MODE (Use `fetch_page_content`, 'interact_with_web'):
   - Use this ONLY when you have reached an "Atomic Leaf Node" (a page dedicated to a single specific entity, like one specific job posting or one specific article).
   - AVOID THE SINGLE-CLICK TRAP: Never call `fetch_page_content` on a directory, a list, or a menu. Use `fetch_page_urls` to drill down until you hit the specific item page.

OPERATIONAL RULES:
- THE ATOMIC DATA RULE: Success is defined by retrieving the specific details of the target (e.g., the job description, not just the job title; the article text, not just the headline).
- THE EXHAUSTION RULE: You must be exhaustive. If the page content suggests a sequence of pages (pagination), you must traverse every page in that sequence. Do not settle for the first page of results.
- LANGUAGE: Prioritize the website's native language pages if they exist, as they often contain more complete information. 
- TERMINATION:
    - SUCCESS: You have found the specific, granular info. Provide the answer clearly, ideally in a structured format, and include the individual source URL(s) for every item found.
    - FAILURE: You have exhausted all logical paths/links and cannot find the granular info. State clearly what you searched for and why the specific data was unreachable.
- ERROR HANDLING: If a tool returns an error (e.g., 404), analyze it and attempt an alternative path or report the failure.
- OUTPUT: Do not respond with text-only plans. Every response must either be a tool call OR the final granular data requested by the user. If you are planning, you must perform the tool call in the same turn.
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

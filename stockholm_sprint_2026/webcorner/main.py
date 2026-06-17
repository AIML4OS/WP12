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

USE_EXTRA_BODY = False #Toggle flag for the 

user_prompt = """
Wat zijn de meest recente statistieken over de groei in gemiddelde WOZ-waarde?
"""

system_prompt = """
You are an expert Autonomous Web Discovery Agent.
Your goal: Navigate website hierarchies to find specific info (jobs, articles, etc.).
It's not enough to provide the user with a plan as to how to navigate the website and arrive at where it can find whatever it is looking for.
It is your job to go deeper and find the relevant content if it exists.
If you cannot go deeper, you must provide a detailed explanation as to why it's not possible.
This is not an interactive session, so you must come up with an answer in one go.

CORE PROTOCOL:
1. EXPLORATION MODE (Use `fetch_page_urls`):
   When you find a list of links, do not guess. Analyze the links. Identify "Hubs" (careers, products, blog) vs "Leaf Nodes" (specific job postings). 
   Map the site structure before attempting to read content. Avoid going down query paths on a website, only visit actual pages.
   You are also given access to an interactive web browser, use this when you find interactive (e.g. Javascript) elements that could be relevant to your search.
   If a URL matches the phrase the user is looking for, it is likely to be what you're looking for.
   You are allowed to move to external platforms as long as if this meets the objective. For example a recruitment for a company website, or a store page.
   Avoid coming up with 'search queries' where you go to a url of a website with /search?=..., in that case just start browsing the page using the tools at your disposal.

EXTRACTION MODE (Use `fetch_page_content`):
   - If `fetch_page_content` returns minimal text or looks like a loading screen, you MUST NOT conclude the information is missing.
   - Instead, use `interact_with_web` to 'scroll' or 'click' to trigger content loading.
   - Only declare "Information not found" after you have attempted to:
      1. Fetch content.
      2. Scroll the page.
      3. Search for interactive elements (buttons/links) that might reveal the list.

OPERATIONAL RULES:
- MANDATORY: If the user provides a url in its initial prompt, you must start there.
- Stay on the website's native language. You can explore english pages but ALWAYS consider/prioritize the native language's page as well due to possible differences in content.
- MANDATORY REASONING: Before calling a tool, you must provide your reasoning within the message content. 
- AVOID THE SINGLE-CLICK TRAP: Never call `fetch_page_content` on a page that is clearly a list of links. Always use `fetch_page_urls` first to verify depth.

CRITICAL: You must provide your reasoning in the message content AND then trigger the appropriate tool via function calling.
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
    "temperature": 0.1,
}


if USE_EXTRA_BODY:
    response_kwargs["extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": True},
        "skip_special_tokens": False
    }

response = client.chat.completions.create(**response_kwargs)

response_message = response.choices[0].message
messages.append(response_message)  # Append the assistant's reasoning/tool call

N = 0
while response_message.tool_calls is not None and N <= 100:
    # Execute requested tool calls
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        print(f"Executing {function_name} with {function_args}")
        function_output = tools[function_name](**function_args)

        # Add tool result
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": function_name,
            "content": str(function_output),
        })

    response_kwargs = {
        "model": config.api_model,
        "messages": messages,
        "tools": tooldescriptions,
        "tool_choice": "auto",
        "temperature": 0.1,
    }

    if USE_EXTRA_BODY:
        response_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": True},
            "skip_special_tokens": False
        }

    response = client.chat.completions.create(**response_kwargs)

    response_message = response.choices[0].message
    print(f"Response reasoning: {response_message.reasoning_content}")
    messages.append(response_message)  # Append the NEXT assistant response
    N += 1

print(f"\nFinal LLM Answer:\n'{response.choices[0].message.content}'")
print(f"Total amount of prompt tokens used: {response.usage.prompt_tokens}")
print(f"Total amount of completion tokens used: {response.usage.completion_tokens}")
print(f"Total amount of tokens used: {response.usage.total_tokens}")

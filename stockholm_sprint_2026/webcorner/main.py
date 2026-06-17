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
Is there a job vacancy related to information security at the swedish statical office? Start on www.scb.se
"""

system_prompt = """
You are an expert Autonomous Web Discovery Agent.
Your goal: Navigate website hierarchies to find specific info (jobs, articles, etc.).

CORE PROTOCOL:
1. EXPLORATION MODE (Use `fetch_page_urls`):
   When you find a list of links, do not guess. Analyze the links. Identify "Hubs" (careers, products, blog) vs "Leaf Nodes" (specific job postings). 
   Map the site structure before attempting to read content. Avoid going down query paths on a website, only visit actual pages. If a URL matches the phrase the user is looking for, it is likely to be what you're looking for.

2. EXTRACTION MODE (Use `fetch_page_content`):
   Only use this when you have reached a "Leaf Node" (a page likely containing the actual answer).

OPERATIONAL RULES:
- MANDATORY: If the user provides a url in its initial prompt, you must start there.
- Stay on the website's native language. You can explore english pages but ALWAYS consider/prioritize the native language's page as well due to possible differences in content.
- MANDATORY REASONING: Before calling a tool, you must provide your reasoning within the message content. 
- AVOID THE SINGLE-CLICK TRAP: Never call `fetch_page_content` on a page that is clearly a list of links. Always use `fetch_page_urls` first to verify depth.

CRITICAL: You must provide your reasoning in the message content AND then trigger the appropriate tool via function calling.
"""
print(f"Starting prompt: {user_prompt}")


# STEP 1: Send the prompt and the tool definition to the LLM
response_kwargs = {
    "model": config.api_model,
    "messages": [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": system_prompt}
    ],
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
tool_calls = response.choices[0].message.tool_calls
messages = [
    {"role": "user", "content": user_prompt}
]

N = 0
while tool_calls is not None and N <= 100:
    # Add response to message history
    messages.append(response_message)

    # Execute requested tool calls
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        function_output = tools[function_name](**function_args)

        # Add the result to history
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
    tool_calls = response_message.tool_calls
    N += 1

print(f"\nFinal LLM Answer:\n'{response.choices[0].message.content}'")
print(f"Total amount of prompt tokens used: {response.usage.prompt_tokens}")
print(f"Total amount of completion tokens used: {response.usage.completion_tokens}")
print(f"Total amount of tokens used: {response.usage.total_tokens}")

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

user_prompt = """
Starting from cbs.nl, I am looking information about import from gulf states, especially recently.
"""

user_prompt = """
Startend vanaf cbs.nl, ik ben op naar informatie over import van de golfstaten, met name recente info.
"""

print(f"Starting prompt: {user_prompt}")


# STEP 1: Send the prompt and the tool definition to the LLM
response = client.chat.completions.create(
    model=config.api_model,  # Or the specific model name used at your lab
    messages=[{"role": "user", "content": user_prompt}],
    tools=tooldescriptions,
    tool_choice="auto"
)

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

    # Get new response given tool call output
    response = client.chat.completions.create(
        model=config.api_model,
        messages=messages,
        tools=tooldescriptions,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    N += 1

print(f"\nFinal LLM Answer:\n'{response.choices[0].message.content}'")

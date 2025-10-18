import openai

# This is a simple test script to check if the API key for the chat service is valid.
# You can also use it to check which models are currently available and use them in
# the main script.
# Use 'path_api_key' variable to specify the path of your api key file
path_api_key = "../api_key.txt"  # "/home/onyxia/work/SCB/Metoddagen/api_key.txt"

# Read API Key
with open(path_api_key, "r", encoding="utf-8") as file:
    api_key = file.readlines()[0]

client = openai.OpenAI(
    api_key=api_key,    # specific for every user
    base_url="https://llm.lab.sspcloud.fr/api"
)

# available:
# bge-m3:latest (no chat support!), gpt-oss:20b, gpt-oss:120b
print('Available models:')
models = client.models.list()
for model in models:
    print('- ' + model.id)

selected_model = 'gpt-oss:20b'
print('\nUsing model "%s"' % (selected_model))
print('- Testing chat session ...')

try:
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert who gives a brief summary of the company's "
                           "activities based on texts I give you."
            },
            {
                "role": "user",
                "content": "Briefly summarize what the company's main business is "
                           "according to the text: We sell pizzas and sushi."
            },
        ],
        temperature=0.2,  # Control the randomness of the response. A value close to 0
                          # makes the output more deterministic, while values closer to 1
                          # add creativity and randomness.
        max_tokens=50,    # Limit the number of tokens
        top_p=0.9         # t only considers the top p probability mass. For instance,
                          # top_p=0.9 would generate tokens from the top 90% probability
                          # distribution.
    )
    print('- Result: ' + response.choices[0].message.content)

except Exception as e:
    print('! Error: %s' % (str(e)))

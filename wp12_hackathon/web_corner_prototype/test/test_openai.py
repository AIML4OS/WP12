import openai

# This is a simple test script to check if the API key for the chat service is valid.
# Use 'path_api_key' variable to specify the path of your api key file
path_api_key = "./api_key.txt"  # "/home/onyxia/work/SCB/Metoddagen/api_key.txt"

# Read API Key
with open(path_api_key, "r", encoding="utf-8") as file:
    api_key = file.readlines()[0]

client = openai.OpenAI(
    api_key=api_key,    # specific for every user
    base_url="https://llm.lab.sspcloud.fr/api"
)

print('Available models:')
models = client.models.list()  # available: bge-m3:latest, gpt-oss:20b, gpt-oss:120b
for model in models:
    print('- ' + model.id)

selected_model = 'gpt-oss:20b'
print('Using model "%s"' % (selected_model))

try:
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": "Du är en expert som ger en kort sammanfattar vad företag har för verksamhet utifrån texter jag ger dig."
            },
            {
                "role": "user",
                "content": "Sammanfatta kort vad företagets huvudsakliga verksamhet är enligt texten: Vi säljer pizzor och sushi."
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
    print(response.choices[0].message.content)

except Exception as e:
    print('! Error: %s' % (str(e)))

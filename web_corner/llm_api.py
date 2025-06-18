import openai

def LLM_API(api_key, input_config = None, model="magistral:24b", message_content=""):
    if len(message_content) < 1:
        error("Message should have content")

    client = openai.OpenAI(
        api_key = api_key,
        base_url = "https://llm.lab.sspcloud.fr/api"
    )

    prompt_content = f"{message_content}, does the above-mentioned text, relating to a web-page, contain a job vacancy posting?"

    response = client.chat.completions.create(
        model = "magistral:24b",
        messages = [
            {"role":"system", "content":"You are an expert about classifying texts, you always give exlusively Yes or No answers, one-word responses."},
            {"role":"user", "content": prompt_content},
        ],
        temperature = 0.2,
        max_tokens = 2,
        top_p = 0.9
    )

    return response.choices[0].message.content

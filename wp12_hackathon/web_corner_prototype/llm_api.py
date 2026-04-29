import openai


def LLM_API(api_key, variables, model="magistral:24b", message_content=""):
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://llm.lab.sspcloud.fr/api"
    )

    prompt_content = f"{message_content}, does the above-mentioned text, "
    "relating to a web-page, contain a {variables}?"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "max_completion_tokens": "1",
                "content": "You are an expert about classifying texts, "
                           "your answer is a one-word response, chosen from [Yes, No]."
            },
            {
                "role": "user",
                "content": prompt_content
            },
        ],
        temperature=0.0,
        max_completion_tokens=1,
        max_tokens=1
    )

    return response.choices[0].message.content

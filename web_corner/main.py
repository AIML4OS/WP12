import requests
import justext
from llm_api import LLM_API

with open("web_corner/api_key.txt", "r", encoding="utf-8") as file:
    api_key = file.readlines()[0]

urls = ... # TODO read urls
urls = [
    "https://www.cbs.nl/nl-nl/vacature/toolexpert/1ae1a790f9284748aca2b34da8cae607",
    "https://www.cbs.nl/nl-nl/visualisaties/dashboard-consumentenprijzen"
]

job_vacancies = {}
for url in urls:
    response = requests.get(url)
    # TODO determine language 
    extracted_paragraphs = justext.justext(response.content, justext.get_stoplist("Dutch"))
    extracted_text = "".join([paragraph.text for paragraph in extracted_paragraphs if not paragraph.is_boilerplate])

    # TODO get prompt from config
    response = LLM_API(api_key, None, "magistral_24b", extracted_text)
    job_vacancies[url] = response

print(job_vacancies)



import requests
import justext
from web_corner.llm_api import LLM_API

urls = ... # TODO read urls
urls = [
    "https://www.cbs.nl/nl-nl/vacature/toolexpert/1ae1a790f9284748aca2b34da8cae607",
    "https://www.cbs.nl/nl-nl/visualisaties/dashboard-consumentenprijzen"
]

job_vacancies = {}
for url in urls:
    response = requests.get(url)
    extracted_text = justext.justext(response.content, justext.get_stoplist("Dutch"))

    # TODO get prompt from config
    response = LLM_API(api_key, None, "magistral_24b", extracted_text)
    job_vacancies[url] = response



# Extract meaningful text
# TODO determine language 

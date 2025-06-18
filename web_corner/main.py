import requests
import justext
from llm_api import LLM_API
import pandas as pd 
import connect_storage
import os
import s3fs
import json
import langcodes
import random

# Create filesystem object
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])
BUCKET = "projet-wp12-url-info-extraction"
FILE_KEY_S3 = "urlsWINChallenge.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    url_data = pd.read_csv(file_in, sep=",")

with open("web_corner/api_key.txt", "r", encoding="utf-8") as file:
    api_key = file.readlines()[0]

url_country = list(zip(url_data['url'], url_data['country']))
random.shuffle(url_country)
url_country = url_country[:500]

job_vacancies = {}
for url, country in url_country:
    try:
        response = requests.get(url)
    except:
        continue
    # TODO determine language 

    language = "English"
    if country == "DE":
        language = "German"
    elif country == "NL":
        language = "Dutch"
    elif country == "PL":
        language = "Polish"
    elif country == "AT":
        country = "German"

    extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
    extracted_text = "".join([paragraph.text for paragraph in extracted_paragraphs if not paragraph.is_boilerplate])

    if len(extracted_text) < 1:
        continue
    # TODO get prompt from config
    response = LLM_API(api_key, None, "magistral_24b", extracted_text)
    job_vacancies[url] = response


print("# of items that are too long")
print(len([v for v in job_vacancies.values() if len(v) > 3]))


with open("web_corner/output.json", "w") as fp:
  json.dump(job_vacancies , fp)
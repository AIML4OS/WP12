import requests
import justext
import os
import s3fs
import json
import random
import pandas as pd

# TODO dit is a hidden script that sets the proper os variable values 
# You can find how to do it on https://datalab.sspcloud.fr/account/storage
# But ideally, you should set these variables when launching the service
import connect_storage
from llm_api import LLM_API


# Create filesystem object
fs = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key = os.environ["AWS_ACCESS_KEY_ID"], 
    secret = os.environ["AWS_SECRET_ACCESS_KEY"], 
    token = os.environ["AWS_SESSION_TOKEN"])
BUCKET = "projet-wp12-url-info-extraction"
FILE_KEY_S3 = "urlsWINChallenge.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3

# Read the URL dataframe containing URL and country code 
with fs.open(FILE_PATH_S3, mode="rb") as file_in:
    url_data = pd.read_csv(file_in, sep=",")

# Read API Key
with open("web_corner/api_key.txt", "r", encoding="utf-8") as file:
    api_key = file.readlines()[0]

# Read the variables for which we want to see if this text contains the content
with open("web_corner/variables.txt", "r", encoding="utf-8") as file:
    variables = file.readlines()

url_country = list(zip(url_data['url'], url_data['country']))
random.shuffle(url_country)

N = 100 # Sample size of list
url_country = url_country[:N]

for var in variables:
    job_vacancies = {}
    print("Variable:", var)
    for url, country in url_country:
        try:
            response = requests.get(url)
        except:
            print("Could not request:", url)
            continue

        # TODO determine language by text content rather than domain
        language = "English"
        if country == "DE":
            language = "German"
        elif country == "NL":
            language = "Dutch"
        elif country == "PL":
            language = "Polish"
        elif country == "AT":
            language = "German"

        if not (response.status_code >= 200 and response.status_code < 400):
            # print(f"Non-viable response status for url: {url}, response: {response.status_code}")
            continue

        # TODO right now we extract text for every variable, but only needs to be done once
        extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
        extracted_text = "".join([paragraph.text for paragraph in extracted_paragraphs if (paragraph.cf_class in ["good", "neargood"] or paragraph.class_type in ["good", "neargood"])])

        if len(extracted_text) < 1:
            # print(f"Issue extracting text for url: {url}, language: {language}")
            continue
        
        response = LLM_API(api_key, var, None, "magistral_24b", extracted_text)
        job_vacancies[url] = response

    print("# of items that are too long")
    print(len([v for v in job_vacancies.values() if len(v) > 3]))
    print(f"Total items: {len(job_vacancies)}/{N}")

    with open(f"web_corner/output_{var.replace(" ", "_").lower()}.json", "w") as fp:
        json.dump(job_vacancies, fp)
    print(job_vacancies)

print("Done!")
import requests
import justext
import s3fs
import json
import random
import pandas as pd

# connect_storage is a script that sets the proper OS variable values.
# You can find how to do it on https://datalab.sspcloud.fr/account/storage
# Ideally, you should set these variables when launching the service
from connect_storage import storage_options

from llm_api import LLM_API

# jusText: is a tool for removing boilerplate content, such as navigation links, headers,
# and footers from HTML pages. It is designed to preserve mainly text containing full
# sentences and it is therefore well suited for creating linguistic resources such as Web corpora.


# Read API Key
def read_api_key(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_key = file.readlines()[0]
        return api_key

    except Exception as e:
        print(f"! Error: could not read the api_key file, {str(e)}")
        print("  Cannot continue execution")
        exit()


# Read the URL dataframe containing URL and country code
def read_url_data(bucket_path, local_path):
    try:
        # Create filesystem object
        fs = s3fs.S3FileSystem(
            client_kwargs={
                'endpoint_url': storage_options['aws_endpoint']
            },
            key=storage_options["aws_access_key_id"],
            secret=storage_options["aws_secret_access_key"],
            token=storage_options["aws_token"])

        with fs.open(bucket_path, mode="rb") as file_in:
            url_data = pd.read_csv(file_in, sep=",")

    except Exception as e:
        print(f'! Error: could not open {bucket_path} from S3, {str(e)}')

        print('Reading data from a local file ...')
        try:
            with open(local_path, "r", encoding="utf-8") as file_in:
                url_data = pd.read_csv(file_in, sep=",")
        except Exception as e:
            print(f'! Error: could not open {local_path}, {str(e)}')
            print("  Cannot continue execution")
            exit()

    return url_data


# Read the variables for which we want to see if this text contains the content
def read_variables(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            variables = file.readlines()
        return variables

    except Exception as e:
        print(f"! Error: could not read the variables file, {str(e)}")
        print("  Cannot continue execution")
        exit()


# Get the language based on country and content
def get_language(country, _content):
    # TODO: determine language by text content rather than from the url domain country
    language = "English"
    if country == "DE":
        language = "German"
    elif country == "NL":
        language = "Dutch"
    elif country == "PL":
        language = "Polish"
    elif country == "AT":
        language = "German"
    return language


# analyze urls
def analyze_urls(sample_size, url_data, variables, api_key, selected_model):
    url_country = list(zip(url_data['url'], url_data['country']))
    random.shuffle(url_country)

    if sample_size != 0:
        url_country = url_country[:sample_size]

    print(f'\nAnalyzing {len(url_country)} urls')
    for var in variables:
        result = {}
        print("\nVariable:", var)
        for url, country in url_country:
            try:
                response = requests.get(url)
            except Exception:
                print("Could not request:", url)
                continue

            if not (response.status_code >= 200 and response.status_code < 400):
                print(f"Non-viable response status for: {url}, status: {response.status_code}")
                continue

            language = get_language(country, response.content)

            # TODO right now we extract text for every variable, but only needs to be done once
            valid_classes = ["good", "neargood"]
            extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
            extracted_text = "".join([
                paragraph.text
                for paragraph in extracted_paragraphs
                if (
                    paragraph.cf_class in valid_classes or
                    paragraph.class_type in valid_classes
                )
            ])

            if len(extracted_text) < 1:
                print(f"Issue extracting text for url: {url}, language: {language}")
                continue

            response = LLM_API(api_key, var, selected_model, extracted_text)
            result[url] = response
            print("Done for %s" % (url))

        num_too_long = len([v for v in result.values() if len(v) > 3])
        print(f"Number of items which analysis answer were too long: {num_too_long}")
        print(f"Total items: {len(result)}/{sample_size}")

        with open(f"output_{var.replace(" ", "_").lower()}.json", "w") as fp:
            json.dump(result, fp)
        print(result)

    print("Done!")


# Setting up parameters ------------------------------------

sample_size = 10  # sample size of urls to analyze, use 0 to analyze all dataset

api_key = read_api_key("./api_key.txt")  # "wp12_hackathon/web_corner_prototype/api_key.txt"

BUCKET = "projet-wp12-url-info-extraction"
FILE_KEY_S3 = "urlsWINChallenge.csv"
FILE_PATH_S3 = BUCKET + "/" + FILE_KEY_S3
url_data = read_url_data(
    FILE_PATH_S3,
    "./test/input.csv")  # specify another local backup input file if necessary
variables = read_variables("./variables.txt")

# use the script 'test/test_openapi.py' to identify available models and
# make sure the selected model supports *chat* mode
selected_model = 'gpt-oss:20b'

# Main execution -------------------------------------------
analyze_urls(
    sample_size,
    url_data,
    variables,
    api_key,
    selected_model)

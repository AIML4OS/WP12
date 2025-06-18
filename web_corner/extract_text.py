import justext
import requests

# Fetch the webpage content
url = "https://www.cbs.nl/nl-nl/vacature/toolexpert/1ae1a790f9284748aca2b34da8cae607"
url = "https://www.cbs.nl/nl-nl/visualisaties/dashboard-consumentenprijzen"
response = requests.get(url)

# Extract meaningful text
# TODO determine language 
paragraphs = justext.justext(response.content, justext.get_stoplist("Dutch"))


# Filter and print the main content
with open("web_corner/output.txt", "w", encoding="utf-8") as file:
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            for line in paragraph.text.split("\n"):
                file.write(line + "\n")


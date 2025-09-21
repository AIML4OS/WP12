import requests
import justext
response = requests.get("https://www.zahnimplantatzentrum.wien/")

print(response.content)

language = "German"
extracted_paragraphs = justext.justext(response.content, justext.get_stoplist(language))
print(extracted_paragraphs[0].__dict__)
print(extracted_paragraphs[0].class_type)

print([paragraph.text for paragraph in extracted_paragraphs if paragraph.class_type in ["good", "neargood"]])
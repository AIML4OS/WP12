# Summarizer

This summarizer is python script that, given a PDF and some parameters, outputs a summary of the document and some metadata (currently tags and keywords).
To be more reusable we aimed at the ability to handle multiple languages.


It allows to test the quality by being able tochoose between several models:

    - **OpenAPI** - you need to provide the key

    - **SSP** - The models on https://llm.lab.sspcloud.fr/, yoou need to get your key from ssplab

    - **Ollama** - if you have an Ollama model running locally

It allows to test the quality of text extraction from PDF using one of the following:

    - **pyPDF**

        - Simpler and more lightweight

        - Part of the LangChain ecosystem

        - Good for basic PDF processing
        
        - May struggle with complex layouts


    - **Docling**

        - More modern and actively maintained

        - Better handling of complex PDF layouts

        - Better support for tables and structured content

        - Can handle more PDF formats and edge cases

        - Generally more robust for production use


    - It might be good to investigate [Unstract](https://github.com/Zipstack/unstract) in terms of licencing, performance of extraction or for inspiration on extraction techniques. 

It allows generating the embeddings remotely, using SSP or Locally by downloading the embedding model from HuggingFace. **Note** Remote use is not working, need to speak with SSP to ask how to invoke embedding api. 



It allows the configuration of the output in the following parameters:

    - Maximum number of words the summary should have

    - Maximum number of keywords to generate

    - Maximum number of tags to generate

    - In what language should the summary be output 


There's quite a bit of room for prompt optimization, but the current verison covers the basic results.

The summarizer_unified.py is the version that combines the vector store(embeddings) and the direct text, thus unifying the prototype_a and prototype_b in terms of pdf processing. 

The output has been transformed to json for easier use. 

## Instructions

Start by installing the dependencies:

````
pip install -r requirements.txt 
````

If you want to run local models, you need to install and setup Ollama first.

The code is designed to be interfaced succintly like this:

````
    summarizer = PDFSummarizer(llm_config=config)
    result = summarizer.process_pdf(
        pdf_path={pdf_file_path},
        max_keywords={number_of_max_keywords},
        max_tags={number_of_max_tags},
        out_lang='{laguange like pt-pt or en}',
        max_words={number_of_max_words}
    )
````

````
    summarizer = PDFSummarizer(llm_config=config)
````
Setup the llm model that will be used for summarization.


````
    def process_pdf(self, pdf_path: str, use_vector_store:bool = False, document_loader:str = "docling", embedding_model:str = "BAAI/bge-m3", ssp_key:str = None, use_remote_embedding:bool = False, max_keywords: int = 6, max_tags: int = 5, out_lang: str = "pt-pt", max_words: int = 200):
````
Call the summarization.


### Parameters

**pdf_path: str** The pdf file to be processed

**document_loader:str = "docling"** Should we use "docling" or "pyPDF"

**use_vector_store:bool = False** Should we use a verctor store or just direct text
    
**use_remote_embedding:bool = True** Should we use SSP to remote process the embedding (uses bge-m3:latest)

**embedding_model:str = "BAAI/bge-m3"** name of the embbeding model (only usefull if use_vector_store=True and use_remote_embedding=False)

**max_keywords: int = 6** Maximum number of Keywords to generate

**max_tags: int = 5** Maximum number of Tags to generate

**out_lang: str = "pt-pt"** Language in wich the summary should be writen

**max_words: int = 200** Maximum number of words that the summary should have


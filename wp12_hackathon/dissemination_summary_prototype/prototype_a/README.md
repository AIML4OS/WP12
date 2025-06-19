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

It allows the configuration of the output in the following parameters:
    - Maximum number of words the summary should have
    - Maximum number of keywords to generate
    - Maximum number of tags to generate
    - In what language should the summary be output 

There's quite a bit of room for prompt optimization, but the current verison covers the basic results.

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

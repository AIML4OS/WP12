from langchain_ollama import OllamaLLM
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from pathlib import Path

# Setting Directories
pdf_dir = Path(__file__).parent
data_dir = pdf_dir/"data"
PDF_CHROMA_DB = str(pdf_dir/"chroma_db")

# Setting the LLM
llm = OllamaLLM(model="llama3.2", temperature=0,top_p=0.9,num_predict=100-150)

# Creating a summary Template
summary_system_template_str = """You are a very efficient, and polite researcher.
Your job is to use summarize 
reports. Use the following context to answer questions.
Be as detailed as possible.

{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=summary_system_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}"
    )
)

messages = [review_system_prompt, review_human_prompt]

review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=messages,
)
my_model_name = "BAAI/bge-m3"#"all-mpnet-base-v2"
sentence_transformer_ef = HuggingFaceEmbeddings(
    model_name=my_model_name
)

pdfs_vector_db = Chroma(
    persist_directory=PDF_CHROMA_DB,
    embedding_function=sentence_transformer_ef
)

reviews_retriever  = pdfs_vector_db.as_retriever(k=5)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | llm
    | StrOutputParser()
)
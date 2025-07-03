from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import requests


# Setting Directories
pdf_dir = Path(__file__).parent
data_dir = pdf_dir/"data"
PDF_CHROMA_DB = str(pdf_dir/"chroma_db")

# Creating sentence tranformer
sentence_transformer_model = "BAAI/bge-m3" #"all-mpnet-base-v2"
sentence_transformer_ef = HuggingFaceEmbeddings(
    model_name=sentence_transformer_model
)

chroma_db = Chroma(
   embedding_function=sentence_transformer_ef, 
   persist_directory=PDF_CHROMA_DB
)

for file in data_dir.iterdir():
    loader = PyPDFLoader(file)
    chunks = loader.load_and_split() # This loads and splits the document into pages
    chroma_db.add_documents(chunks) # Not efficient, but I have CUDA memory limitations.
    response = requests.post("http://127.0.0.1:8000/upload-pdf/", json={"list_docs":chunks})
print(f"ChromaDB with local embeddings created")
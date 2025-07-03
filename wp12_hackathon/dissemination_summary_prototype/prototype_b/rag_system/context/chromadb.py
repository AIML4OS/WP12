from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_core.vectorstores import VectorStoreRetriever


class ChromaLangDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # "BAAI/bge-m3", or "all-MiniLM-L6-v2"
        self.sentence_model = model_name
        self.vectorDB: Chroma
    
    @property
    def embedding_function(self):
        return HuggingFaceEmbeddings(model_name=self.sentence_model)
    
    def create_db(self):
        self.vectorDB = Chroma(embedding_function=self.embedding_function)

    def send_to_db(self, pages: list[str]):
        documents = []
        for page in pages:
            documents.append(Document(page_content=page)) 
        self.vectorDB.add_documents(documents)

    def context_builder(self, num_docs:int) -> VectorStoreRetriever:
        return self.vectorDB.as_retriever(k=num_docs)
    
    def similarity_search(self, question:str, **kwargs) -> list:
        return self.vectorDB.similarity_search(query=question, **kwargs)

 
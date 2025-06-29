from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class ChromaLangDB:
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.sentence_model = model_name
        self.db: Chroma | None = None
    
    @property
    def embedding_function(self):
        return HuggingFaceEmbeddings(model_name=self.sentence_model)
    
    def create_vectorDB(self):
        self.vectorDB = Chroma(embedding_function=self.embedding_function)

    def send_to_vectorDB(self, documents):
        self.vectorDB.add_documents(documents)

    def get_context(self, num_docs:int):
        self.vectorDB.as_retriever(k=num_docs)
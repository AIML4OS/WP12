from rag_system.input.pdf import PdfReaderInput
from typing import BinaryIO
from rag_system.context.chromadb import ChromaLangDB

class RAGSystem:
    def __init__(self, input:PdfReaderInput, db: ChromaLangDB) -> None:
        self.input = input
        self.db = db
        self.db.create_db()

    def read_io(self, content: BinaryIO):
        self.input.parse(content)

    def send_to_db(self):
        self.db.send_to_db(self.input.pages)

    def get_context(self, num_docs: int):
        return self.db.get_context(num_docs)
    
    def summarize(self) -> str:
        """To be rewritten!"""
        question = """Qual o movimento dos aeroportos nacionais em Abril?"""
        relevant_docs = self.db.similarity_search(question, k=3)
        return relevant_docs[1].page_content
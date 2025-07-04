from typing import Protocol,runtime_checkable, Any

@runtime_checkable
class VectorDBBase(Protocol):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.sentence_model:str
        self.db: Any
    
    @property
    def embedding_function(self)-> Any:...
    
    def create_db(self):...
        
    def send_to_db(self, pages:list[str]):...

    def get_context(self, num_docs:int) -> Any:...

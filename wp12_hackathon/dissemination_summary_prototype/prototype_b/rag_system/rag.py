from rag_system.input.pdf import PdfReaderInput
from typing import BinaryIO, Any
from rag_system.context.chromadb import ChromaLangDB
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rag_system.prompt_template.langchain_template import LangChainPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from rag_system.model.ollama import Ollama


class RAGSystem:
    def __init__(self, input:PdfReaderInput, 
                 db: ChromaLangDB, 
                 prompt_builder: LangChainPromptTemplate,
                 model: Ollama,
                 topic: str = "",
                 human_prompting: bool = True,
                 keywords: int = 5,
                 tags: int = 5,
                 max_words: int = 100,
                 out_lang: str = "pt-pt") -> None:
        self.input = input
        self.db = db
        self.db.create_db()
        self.prompt_builder = prompt_builder
        self.model = model
        self.keywords = keywords
        self.tags = tags
        self.max_words = max_words
        self.out_lang = out_lang
        self.topic=topic

    def read_io(self, content: BinaryIO):
        self.input.parse(content)

    def send_to_db(self):
        self.db.send_to_db(self.input.pages)

    def context_builder(self, num_docs: int) -> VectorStoreRetriever:
        return self.db.context_builder(num_docs)
    
    def prompt_template(self, human_prompting:bool):
        return self.prompt_builder.create_prompt(human_prompting)
    
    def summarize(self, 
                  num_docs: int = 5, 
                  human_prompting: bool = True,
                  keywords: int | None = None,
                  tags: int | None = None,
                  max_words: int | None = None,
                  out_lang: str | None = None
                ) -> Any:
        """
        Here's a summary

        Arguments
        ---------
        keywords
        """
        if keywords:
            self.keywords = keywords
        if tags:
            self.tags = tags
        if max_words:
            self.max_words = max_words
        if out_lang:
            self.out_lang = out_lang
        context_retriever = self.context_builder(num_docs)
        prompt = self.prompt_template(human_prompting)
        self.chain = prompt | self.model.core | StrOutputParser()
        reports = ""
        context_docs = context_retriever.invoke(self.topic)
        for doc in context_docs:
            reports+= "\n" + doc.page_content
        res =  self.chain.invoke({
            "reports": reports,
            "keywords": self.keywords,
            "tags": self.tags,
            "max_words":self.max_words,
            "out_lang": self.out_lang,
        })
        print(res)
        return res
         
        
    # async def invoke_system(summary_request: str) -> SummaryRequestOutput:
#     """
#     Invoke the system to return a summary of the pdf.
#     """
#     summary = 
#     return SummaryRequestOutput(output=summary)
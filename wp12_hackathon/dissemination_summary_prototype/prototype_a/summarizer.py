import os
from typing import List, Dict, Optional, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter

from dotenv import load_dotenv
import logging

########################################################################################
#
# OLD CODE 
#
# This is the initial version of the summarizer. 
# The relevant code of this file has been ported to the summarizer_unified.py file 
# where it was merged with the vector store approach from prototype b.
#
########################################################################################

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

class PDFSummarizer:
    def __init__(
        self,
        llm_provider: str = "ssp",
        llm_config: Dict = None,
        max_keywords: int = 6,
        max_tags: int = 5,
        max_words: int = 200,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        use_docling: bool = True,
        out_lang: str = "pt-pt", 
        
    ):
        self.max_keywords = max_keywords
        self.max_tags = max_tags
        self.max_words = max_words
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_docling = use_docling
        self.out_lang = out_lang
       
        # Initialize LLM with configuration
        llm_config = llm_config or {}
        self.llm = self._create_llm(llm_provider, **llm_config)
        
        # Inner system prompt (fixed)
        self.inner_system_prompt = """
        You are a professional document analyzer. Your task is to:
        1. Provide a clear and concise summary
        2. Extract relevant keywords and tags
        3. Maintain objectivity and accuracy
        4. Focus on the main points and key findings
        """
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Initialize docling converter if enabled
        if self.use_docling:
            try:
                self.docling_converter = DocumentConverter()
            except Exception as e:
                logger.warning(f"Failed to initialize docling: {e}. Falling back to PyPDFLoader.")
                self.use_docling = False

    def _create_llm(self, provider: str = "ssp", **kwargs) -> BaseChatModel:
        """
        Create LLM instance based on provider.
        """
        if provider.lower() == "openai":
            return ChatOpenAI(
                model_name=kwargs.get("model_name", "gpt-3.5-turbo"),
                temperature=kwargs.get("temperature", 0.7)
            )
        elif provider.lower() == "ollama":
            return ChatOllama(
                model="llama3.1:8b",
                temperature=kwargs.get("temperature", 0.7),
                base_url=kwargs.get("base_url", "http://localhost:11434")
            )
        elif provider.lower() == "ssp":
            #model = "mistral-small3.1:latest",
            #model = "llama3.3:70b",

            return ChatOpenAI(
                api_key = kwargs.get("api_key"),  # replace with your key
                base_url="https://llm.lab.sspcloud.fr/api",
                model="mistral-small3.1:latest",
                temperature=kwargs.get("temperature", 0.7)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _load_pdf_with_docling(self, pdf_path: str) -> str:
        """
        Load PDF using docling's advanced parsing capabilities.
        """
        try:
            # Convert the PDF using docling
            result = self.docling_converter.convert(pdf_path)
            
            # Extract text content with proper reading order
            text_content = []
            
            # Get text from the document body
            for text_item in result.document.texts:
                if text_item.text and text_item.text.strip():
                    text_content.append(text_item.text.strip())
            
            # Join all text content with proper spacing
            return "\n\n".join(text_content)
            
        except Exception as e:
            raise e

    def _load_pdf_with_pypdf(self, pdf_path: str) -> str:
        """
        Fallback method to load PDF using PyPDFLoader.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _create_summary_chain(self, custom_system_prompt: str):
        """
        Create a runnable sequence for summarization.
        """
        system_message = f"{self.inner_system_prompt}\n\n{custom_system_prompt}\n\n" + \
            "Format your response in the following structure:\n" + \
            "SUMMARY:\n<comprehensive summary here>\n\n" + \
            "KEYWORDS:\n<comma-separated keywords>\n\n" + \
            "TAGS:\n<comma-separated tags>"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Please analyze the following text and provide:\n"
                     "1. A comprehensive summary\n"
                     "2. {max_keywords} most important keywords\n"
                     "3. {max_tags} relevant tags\n\n"
                     "4. With a maximum of {max_words} words"
                     "5. Provide the output in the following language: {out_lang}"
                     "Text: {text}")
        ])
        
        return prompt | self.llm | StrOutputParser()

    def process_pdf(
        self,
        pdf_path: str,
        # custom_system_prompt: str,
        max_keywords: Optional[int] = 6,
        max_tags: Optional[int] = 5,
        max_words: Optional[int] = 200,
        out_lang: Optional[str] = 'pt-pt',
    ) -> Dict:
        """
        Process a PDF file and return summary, keywords, and tags.
        """
        # Override default limits if provided
        if max_keywords:
            self.max_keywords = max_keywords
        if max_tags:
            self.max_tags = max_tags
        if max_words:
            self.max_words = max_words
        if out_lang:
            self.out_lang = out_lang

        # Load PDF using docling if enabled, otherwise fallback to PyPDFLoader
        if self.use_docling:
            text = self._load_pdf_with_docling(pdf_path)
        else:
            text = self._load_pdf_with_pypdf(pdf_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)

        context = "\n\n=== Document Section ===\n\n".join(chunks)

        # Process each chunk
        instruction_prompt = """
    You are a statiticitian, specialized in reporting, you goal is to summarize the content of documents.
    You don't add information and you don't make analysis, you just summarize what is already there.
    You pay special attention not to be biased in your summarizations, stay neutral and follow the spirit and tone present in the document.
    When outputing in Portuguese, use the European Portuguese.
    """
  
        self._create_summary_chain(instruction_prompt)
        
        # Extract final keywords and tags from combined summary
        final_chain = self._create_summary_chain(
            "Please provide ONLY the final list of keywords and tags based on the entire document."
        )
        final_result = final_chain.invoke({
            "text": context,
            "max_keywords": self.max_keywords,
            "max_tags": self.max_tags,
            "max_words": self.max_words,
            "out_lang": self.out_lang
        })

        return {
            "summary": final_result
        }


def main():
    config = {
        "model": 'mistral-small3.1:latest',
        "api_key": os.getenv('SSP_KEY'),
    }
    # model = "mistral-small3.1:latest",
    # model = "llama3.3:70b",

    summarizer = PDFSummarizer(llm_config=config)
    result = summarizer.process_pdf(
        pdf_path="Aereo.pdf",
        max_keywords=15,
        max_tags=8,
        out_lang='en',   # 'pt-pt'
        max_words=200
    )

    print(result["summary"])

if __name__ == "__main__":
    main() 
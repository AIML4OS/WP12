# this is a unified summarizer that has parameters for the different summarizers

# Load environment variables FIRST, before any other imports
from dotenv import load_dotenv
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Load .env from the script's directory, not the current working directory
load_dotenv(os.path.join(script_dir, '.env'))


import warnings
import time
from typing import List, Dict, Optional, Union
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_docling.loader import ExportType
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
import torch
import logging
import json
from langchain_community.vectorstores.utils import filter_complex_metadata

from langchain_core.embeddings import Embeddings
import requests

# Suppress PyTorch MPS warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

# Configure logging - reduce verbosity
logging.basicConfig(level=logging.WARNING)  # Changed from INFO to WARNING
logger = logging.getLogger(__name__)


def timed_execution(message_template: str = None):
    """
    Decorator that times function execution and prints a custom message with timing.
    
    Args:
        message_template (str): Template message to format with function parameters. 
                               If None, uses the function name.
                               Can use parameter names in curly braces like {pdf_path}
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Format message template with actual parameters if provided
            if message_template:
                try:
                    # For instance methods, args[0] is self, so we need to get the actual parameters
                    if args and hasattr(args[0], '__class__'):  # Instance method
                        # Get function signature to map parameters
                        import inspect
                        sig = inspect.signature(func)
                        bound_args = sig.bind(*args, **kwargs)
                        bound_args.apply_defaults()
                        
                        # Create a dict with parameter names and values
                        param_dict = dict(bound_args.arguments)
                        # Remove 'self' from the dict
                        if 'self' in param_dict:
                            del param_dict['self']
                        
                        display_message = message_template.format(**param_dict)
                    else:  # Regular function
                        display_message = message_template.format(*args, **kwargs)
                except (KeyError, IndexError) as e:
                    # Fallback if formatting fails
                    display_message = f"Executing {func.__name__} (format error: {e})"
            else:
                display_message = f"Executing {func.__name__}"
            
            print(f"{display_message}")
            
            # Time the execution
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Print timing
            print(f"⏱️ Execution time: {execution_time:.2f} seconds")
            return result
        return wrapper
    return decorator


def execute_with_timing(func, message: str = None, *args, **kwargs):
    """
    Execute a function with timing and custom message.
    
    Args:
        func: Function to execute
        message (str): Custom message to print before execution
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        The result of the function execution
    """
    # Use custom message or function name
    display_message = message or f"Executing {func.__name__}"
    print(f"{display_message}")
    
    # Time the execution
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    
    # Print timing
    print(f"⏱️ Execution time: {execution_time:.2f} seconds")
    return result


class PDFSummarizer:
    def __init__(
        self,
        llm_provider: str = "ssp",
        llm_config: Dict = None
    ):
        # Initialize LLM with configuration
        llm_config = llm_config or {}
        self.llm = self._create_llm(llm_provider, **llm_config)

    def _safe_json_parse(self, result):
        """Parse JSON or return text if parsing fails."""
        import re
        
        content = result.content if hasattr(result, 'content') else str(result)
        
        # First try direct JSON parsing
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object anywhere in the text
        json_match = re.search(r'\{.*?\}', content, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If all fails, return fallback
        print(f"⚠️ JSON parsing failed completely")
        print(f"🔍 Raw LLM output: {content[:200]}...")
        return {
            "summary": content,
            "keywords": ["parsing_failed"],
            "tags": ["raw_text"]
        }

    def _create_llm(self, provider: str, **kwargs) -> BaseChatModel:
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
                model=kwargs.get("model", "llama3.1:8b"),
                temperature=kwargs.get("temperature", 0.7),
                base_url=kwargs.get("base_url", "http://localhost:11434")
            )
        elif provider.lower() == "ssp":
            #model = "mistral-small3.1:latest",
            #model = "llama3.3:70b",

            return ChatOpenAI(
                api_key = kwargs.get("api_key"),  # replace with your key
                base_url="https://llm.lab.sspcloud.fr/api",
                model=kwargs.get("model", "mistral-small3.2:latest"),
                temperature=kwargs.get("temperature", 0.7)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @timed_execution("Processing PDF: {pdf_path}, use_vector_store: {use_vector_store}, document_loader: {document_loader}, embedding_model: {embedding_model}, use_remote_embedding: {use_remote_embedding}")
    def process_pdf(self, pdf_path: str, use_vector_store:bool = False, document_loader:str = "docling", embedding_model:str = "BAAI/bge-m3", use_remote_embedding:bool = False, max_keywords: int = 6, max_tags: int = 5, out_lang: str = "pt-pt", max_words: int = 200):
        """
        Process a PDF file and return a summary.
        """

        # - 1 - load the pdf with docling or pypdf
        # - 2 - split the pdf into chunks
        # - 3 - load the chunks into a vector store or just the text        
        # - 4 - summarize the chunks with the llm (querying the vector store) or (just the text)
        # - 5 - return the summary

        if use_vector_store:
            if document_loader == "docling":
                chunks = self._load_pdf_with_docling(pdf_path)
            elif document_loader == "pypdf":
                chunks = self._load_pdf_with_pypdf(pdf_path)
            else:
                raise ValueError(f"Unsupported document loader: {document_loader}")
            
            vector_store = self._get_vector_store(chunks, embedding_model, use_remote_embedding)
            retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Increased from 5 to 10

            # Create retrieval chain with integrated prompting
            retrieval_chain = self._create_retrieval_chain(retriever)
            
            result = retrieval_chain.invoke({
                "input": "Summarize the main content and statistics from this document about air transport activity",
                "max_keywords": max_keywords,
                "max_tags": max_tags,
                "max_words": max_words,
                "out_lang": out_lang
            })
            # Return the JSON result directly
            return result.get("answer", result)
        
        else:
            context = "\n\n=== Document Section ===\n\n"
            if document_loader == "docling":
                pdf_text = self._load_pdf_txt_with_docling(pdf_path)
                context = context + pdf_text
            elif document_loader == "pypdf":
                pdf_text = self._load_pdf_txt_with_pypdf(pdf_path)
                context = context + pdf_text
            else:
                raise ValueError(f"Unsupported document loader: {document_loader}")

            # 3 - create the summary chain
            chain = self._create_summary_chain()

            result = chain.invoke({
                "content": context,
                "max_keywords": max_keywords,
                "max_tags": max_tags,
                "max_words": max_words,
                "out_lang": out_lang
            })

            return result


    def _create_summary_chain(self):
        """
        Create a runnable sequence for direct text summarization (non-vector store mode).
        """
        chat_prompt, _ = self._get_shared_prompts()
        return chat_prompt | self.llm | self._safe_json_parse


    def _load_pdf_txt_with_docling(self, pdf_path: str) -> str:
        """
        Load PDF using docling's advanced parsing capabilities.
        """
        try:
            # Temporarily suppress logging during conversion
            with self._suppress_logging():
                # Convert the PDF using docling
                docling_converter = DocumentConverter()
                result = docling_converter.convert(pdf_path)
            
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

    def _load_pdf_txt_with_pypdf(self, pdf_path: str) -> str:
        """
        Fallback method to load PDF using PyPDFLoader.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n".join(page.page_content for page in pages)

    def _load_pdf_with_docling(self, pdf_path: str, sentence_transformer_model: str = "BAAI/bge-m3") -> List[str]:
        """
        Load PDF using docling's advanced parsing capabilities.
        """
        export_type = ExportType.DOC_CHUNKS
        try:
            # Temporarily suppress logging during conversion
            with self._suppress_logging():
                # Convert the PDF using docling
                loader = DoclingLoader(
                    file_path=pdf_path,
                    export_type=export_type,
                    chunker=HybridChunker(tokenizer=sentence_transformer_model),
                )

                docs = loader.load()
                # split the docs into chunks
                if export_type == ExportType.DOC_CHUNKS:
                    splits = docs
                    # Filter complex metadata from docling documents
                    splits = filter_complex_metadata(splits)

                elif export_type == ExportType.MARKDOWN:
                    from langchain_text_splitters import MarkdownHeaderTextSplitter

                    splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[
                            ("#", "Header_1"),
                            ("##", "Header_2"),
                            ("###", "Header_3"),
                        ],
                    )
                    splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]
                else:
                    raise ValueError(f"Unexpected export type: {export_type}")
                
                return splits
            

        except Exception as e:
            raise e

    def _get_vector_store(self, splits: List[str], embedding_model: str = "BAAI/bge-m3", use_remote_embedding: bool = True) -> str:
        """
        Create vector store with local or remote embeddings.
        
        Args:
            splits: Document chunks
            embedding_model: Model name for embeddings
            use_remote_embedding: If True, use remote embedding service
        """
        if use_remote_embedding:
            # Use remote embedding service (SSP BGE-M3)
            remote_model = "bge-m3:latest"  # model name for SSP API

            print(f"🔧 Remote embedding with model: {remote_model} at SSP")
            
            #Configure for SSP's embedding service
            embedding = OllamaEmbeddings(
                model=remote_model,
                base_url="http://llm.lab.sspcloud.fr/ollama",
                client_kwargs={'headers': {
                    "Authorization": f"Bearer {os.getenv('SSP_KEY')}"
                }}
            )

        else:
            # Use local embedding
            print(f"🔧 Local embedding with model: {embedding_model}")

            embedding = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                encode_kwargs={"normalize_embeddings": True}  # recommended for BGE
            )

        vector_store = Chroma.from_documents(
            persist_directory="chroma_db",
            documents=splits,
            embedding=embedding
        )
        return vector_store

    def _load_pdf_with_pypdf(self, pdf_path: str) -> str:
        """
        Load PDF using pypdf with better chunking strategy.
        """
        from langchain_core.documents import Document
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Use RecursiveCharacterTextSplitter for better chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks
            chunk_overlap=200,  # Some overlap for context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the pages into better chunks and create Document objects
        splits = []
        for page in pages:
            page_splits = text_splitter.split_text(page.page_content)
            for split in page_splits:
                # Create Document objects with metadata
                doc = Document(
                    page_content=split,
                    metadata=page.metadata
                )
                splits.append(doc)
        
        return splits

    # this is a helper function to suppress the logging of docling, it is used to avoid the verbose output of docling   
    def _suppress_logging(self):
        """
        Context manager to temporarily suppress logging.
        """
        import contextlib
        
        @contextlib.contextmanager
        def suppress_logging():
            # Store original levels
            original_levels = {}
            loggers_to_suppress = [
                "docling.document_converter",
                "docling.models.factories", 
                "docling.utils.accelerator_utils",
                "docling.pipeline.base_pipeline",
                "torch.utils.data.dataloader",
                "chromadb.telemetry.product.posthog",
                "chromadb.telemetry"
            ]
            
            try:
                # Set all to ERROR level (suppress INFO and WARNING, keep ERROR)
                for logger_name in loggers_to_suppress:
                    logger_obj = logging.getLogger(logger_name)
                    original_levels[logger_name] = logger_obj.level
                    logger_obj.setLevel(logging.ERROR)
                yield
            finally:
                # Restore original levels
                for logger_name, original_level in original_levels.items():
                    logging.getLogger(logger_name).setLevel(original_level)
        
        return suppress_logging()

    def _create_retrieval_chain(self, retriever):
        """
        Create a retrieval chain that integrates document retrieval with generation.
        """
        _, string_prompt = self._get_shared_prompts()
        
        # Create the document chain with safe JSON output
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=string_prompt,
            output_parser=self._safe_json_parse
        )
        
        # Create the retrieval chain
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
        
        return retrieval_chain

    def _get_shared_prompts(self):
        """
        Get shared prompts that ensure consistent output format across all modes.
        """
        # Instruction prompt (this is the one we can tweak, it could be refactored as a parameter)
        instruction_prompt = """
        You are a statistician, specialized in reporting, your goal is to summarize the content of documents.
        You don't add information and you don't make analysis, you just summarize what is already there.
        You pay special attention not to be biased in your summarizations, stay neutral and follow the spirit and tone present in the document.
        When outputting in Portuguese, use the European Portuguese.
        """
        
        # Inner system prompt (this is fixed)
        inner_system_prompt = """
        You are a professional document analyzer. Your task is to:
        1. Provide a clear and concise summary
        2. Extract relevant keywords and tags
        3. Maintain objectivity and accuracy
        4. Focus on the main points and key findings

        CRITICAL: You must respond ONLY with valid JSON. No explanations, no markdown, no additional text - just pure JSON.
        """

        system_message = f"{inner_system_prompt}\n\n{instruction_prompt}\n\n"
        
        # Single human prompt template to avoid duplication
        human_prompt = """You must analyze the content and return ONLY valid JSON. No explanations, no markdown, no additional text.

Required JSON structure:
{{
    "summary": "comprehensive summary (max {max_words} words, language: {out_lang})",
    "keywords": ["list of {max_keywords} keywords"],
    "tags": ["list of {max_tags} tags"]
}}

Example valid response:
{{
    "summary": "The document discusses air transport statistics showing passenger growth.",
    "keywords": ["aviation", "statistics", "passengers"],
    "tags": ["transport", "data"]
}}

Now analyze this content and respond with JSON only:

{content}"""
        
        # Chat prompt template (for direct text mode)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_prompt)
        ])
        
        # String prompt template (for retrieval chain) - uses same human prompt with different variable names
        retrieval_human_prompt = human_prompt.replace("{content}", "Context: {context}\n\nUser Question: {input}")
        string_prompt = PromptTemplate(
            template=system_message + retrieval_human_prompt,
            input_variables=["context", "input", "max_keywords", "max_tags", "max_words", "out_lang"]
        )
        
        return chat_prompt, string_prompt


def main():
        
    # config for the llm
    config = {
        "api_key": os.getenv('SSP_KEY'),
        "temperature": 0.1  # Lower temperature for more deterministic JSON output
    }
    # model = "mistral-small3.1:latest", # a bit faster
    # model = "llama3.3:70b",   # a bit better but slower


    # 1 - create and configure the summarizer
    summarizer = PDFSummarizer(llm_config=config)
    
    # our input file - make it relative to the script's directory
    file_path = os.path.join(script_dir, "Aereo.pdf")

    # result = summarizer.process_pdf(
    #     file_path,
    #     document_loader="pypdf",
    #     use_vector_store=True,
    #     #     embedding_model="thenlper/gte-small", # alternative small fast and weak model for local embeddings
    # )
    # print(json.dumps(result, indent=2, ensure_ascii=False))
    # print("\n" + "="*80 + "\n")

    result = summarizer.process_pdf(
        file_path,
        use_vector_store=True,
        use_remote_embedding=True
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + "="*80 + "\n")


    # example of use
    # result = summarizer.process_pdf(
    #     pdf_path="Aereo.pdf",
    #     use_vector_store=False,
    #     document_loader="docling",
    #     max_keywords=15,
    #     max_tags=8,
    #     out_lang='en',   # 'pt-pt'
    #     max_words=200
    # )



if __name__ == "__main__":
    main() 
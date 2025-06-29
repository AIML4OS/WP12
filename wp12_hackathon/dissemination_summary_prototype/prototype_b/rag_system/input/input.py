from langchain_community.document_loaders import PyPDFLoader
import PyPDF2
from rag_system.input.input_base import InputParserBase
from typing import Any

from pathlib import Path

class PyPDF2Parser:
    def __init__(self) -> None:
        self.parser = PyPDF2.PdfReader
        self.parsed_content: Any = None

    def parse(self, input_docs: Any) -> None:
        self.parsed_content = self.parser(input_docs)

class PyPDFLoaderParser:
    def __init__(self) -> None:
        self.parser = PyPDFLoader
        self.parsed_content: Any = None

    def parse(self, input_docs: Path) -> None:
        loader = self.parser(input_docs)
        self.parsed_content = loader.load_and_split()

def validateInputBase(obj: InputParserBase) -> bool:
    """Runtime validation that an object implements InputParserBase protocol"""
    return isinstance(obj, InputParserBase)


if __name__ == "__main__":
    print(validateInputBase(PyPDFLoaderParser()))   # True
    print(validateInputBase(PyPDF2Parser()))  # True
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfReader
from rag_system.input.input_base import InputBase
from typing import BinaryIO

from pathlib import Path


class PdfReaderInput:
    def __init__(self) -> None:
        self.parser = PdfReader
        self.parsed_content: PdfReader | None = None

    def parse(self, input_docs: BinaryIO) -> None:
        self.parsed_content = self.parser(input_docs)

    @property
    def pages(self)-> list[str]:
        if self.parsed_content:
            return [page.extract_text() for page in self.parsed_content.pages]
        raise TypeError("parsed_content cannot")


class PyPDFLoaderInput:
    def __init__(self) -> None:
        self.parser = PyPDFLoader
        self.parsed_content: PyPDFLoader | None = None

    def parse(self, input_docs: Path) -> None:
        loader = self.parser(input_docs)
        self.parsed_content = loader.load_and_split() # type: ignore




if __name__ == "__main__":
    def validateInputBase(obj: InputBase) -> bool:
        """Runtime validation that an object implements InputParserBase protocol"""
        return isinstance(obj, InputBase)
    print(validateInputBase(PdfReaderInput()))   # True
    print(validateInputBase(PyPDFLoaderInput()))  # True
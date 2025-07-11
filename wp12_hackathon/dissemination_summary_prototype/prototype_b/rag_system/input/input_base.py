from typing import Protocol,runtime_checkable, Any

@runtime_checkable
class InputBase(Protocol):
    parser: Any
    parsed_content: Any
    
    def parse(self, input_docs: Any) -> None:
        ...



from pydantic import BaseModel

class SummaryRequestInput(BaseModel):
    text: str

class SummaryRequestOutput(BaseModel):
    output: str
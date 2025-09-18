from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class ExtractedData:
    company_name: str
    report_year: int
    financial_metrics: Dict[str, Any]
    operational_metrics: Dict[str, Any]
    esg_metrics: Dict[str, Any]
    raw_text_sample: str


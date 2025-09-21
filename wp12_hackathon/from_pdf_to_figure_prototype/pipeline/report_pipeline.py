from datetime import datetime
from typing import Dict

from processors.text_processor import FinalPDFProcessor
from extractors.base_extractor import EnhancedAnnualReportExtractor
from validators.data_validator import DataValidator


class UpdatedAnnualReportPipeline:
    def __init__(self, api_key: str, base_url: str, model_id: str):
        self.pdf_processor = FinalPDFProcessor()  # Your OCR-enabled processor
        self.extractor = EnhancedAnnualReportExtractor(api_key, base_url, model_id)
        self.validator = DataValidator()

    def process_report(self, pdf_path: str) -> Dict:
        """Complete pipeline with OCR support"""
        print(f"Processing: {pdf_path}")

        # Step 1: Extract text (will use OCR if needed)
        text, extraction_metadata = self.pdf_processor.extract_text(pdf_path)

        print(f"Extraction method: {extraction_metadata['method_used']}")
        print(f"Text length: {len(text)} characters")

        # Skip if extraction failed completely
        if extraction_metadata['extraction_quality'] == 'poor':
            return {
                "source_file": pdf_path,
                "status": "failed",
                "reason": "Poor text extraction even with OCR",
                "metadata": extraction_metadata
            }

        # Step 2: Extract structured data with LLM
        file_name = pdf_path.split('/')[-1]
        raw_extraction = self.extractor.extract_structured_data(text, file_name)

        # Step 3: Validate results
        validated_data = self.validator.validate_and_clean(raw_extraction)

        return {
            "source_file": pdf_path,
            "processing_timestamp": datetime.now().isoformat(),
            "extraction_metadata": extraction_metadata,
            "extracted_data": validated_data,
            "status": "success" if "error" not in validated_data else "partial"
        }

import pytesseract
from PIL import Image
import fitz
import io
from typing import Tuple, Dict
import PyPDF2

class ImprovedPDFProcessor:
    def extract_text(self, pdf_path: str, max_pages: int = 50) -> Tuple[str, dict]:
        """Extract text with metadata about extraction quality"""
        text = ""
        metadata = {
            "method_used": None,
            "pages_processed": 0,
            "extraction_quality": "unknown"
        }

        try:
            # Try PyMuPDF first (more robust)
            doc = fitz.open(pdf_path)
            pages_to_process = min(len(doc), max_pages)

            for page_num in range(pages_to_process):
                page = doc[page_num]
                page_text = page.get_text()
                text += page_text + "\n"

            doc.close()
            metadata["method_used"] = "PyMuPDF"
            metadata["pages_processed"] = pages_to_process

            # Check extraction quality
            if len(text.strip()) > 1000:
                metadata["extraction_quality"] = "good"
            elif len(text.strip()) > 100:
                metadata["extraction_quality"] = "moderate"
            else:
                metadata["extraction_quality"] = "poor"

        except Exception as e:
            print(f"PyMuPDF failed: {e}, falling back to PyPDF2")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages_to_process = min(len(pdf_reader.pages), max_pages)

                    for page_num in range(pages_to_process):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n"

                metadata["method_used"] = "PyPDF2"
                metadata["pages_processed"] = pages_to_process
                metadata["extraction_quality"] = "moderate" if len(text) > 100 else "poor"

            except Exception as e2:
                metadata["method_used"] = "failed"
                metadata["error"] = str(e2)

        return text[:50000], metadata


class ImagePDFProcessor:
    def extract_text_with_ocr(self, pdf_path: str, max_pages: int = 50) -> Tuple[str, dict]:
        """Extract text from image-based PDFs using OCR"""
        text = ""
        metadata = {
            "method_used": "OCR",
            "pages_processed": 0,
            "extraction_quality": "poor"
        }

        try:
            doc = fitz.open(pdf_path)
            pages_to_process = min(len(doc), max_pages)

            for page_num in range(pages_to_process):
                page = doc.load_page(page_num)

                # Convert page to high-resolution image for better OCR
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))

                # Apply OCR
                page_text = pytesseract.image_to_string(
                    image,
                    lang='eng',
                    config='--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
                )

                text += page_text + "\n"
                print(f"OCR Page {page_num + 1}: extracted {len(page_text)} characters")

            doc.close()
            metadata["pages_processed"] = pages_to_process

        except Exception as e:
            print(f"OCR extraction failed: {e}")
            metadata["error"] = str(e)

        # Assess quality
        if len(text.strip()) > 1000:
            metadata["extraction_quality"] = "good"
        elif len(text.strip()) > 100:
            metadata["extraction_quality"] = "moderate"

        return text[:50000], metadata


class FinalPDFProcessor:
    def extract_text(self, pdf_path: str, max_pages: int = 50) -> Tuple[str, dict]:
        """Complete extraction pipeline with OCR fallback"""
        text = ""
        metadata = {"method_used": "failed"}

        # Step 1: Try standard extraction
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), max_pages)):
                page_text = doc.load_page(page_num).get_text()
                text += page_text + "\n"
            doc.close()

            if len(text.strip()) > 50:
                metadata["method_used"] = "standard_extraction"
                metadata["extraction_quality"] = "good"
                return text[:50000], metadata

        except Exception as e:
            print(f"Standard extraction failed: {e}")

        # Step 2: Use OCR for image-based PDFs
        print("Switching to OCR extraction...")
        ocr_processor = ImagePDFProcessor()
        return ocr_processor.extract_text_with_ocr(pdf_path, max_pages)
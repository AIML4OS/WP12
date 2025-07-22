"""
Industry-intelligent document analyzer
core/industry_analyzer.py
"""

import pdfplumber
import re
from typing import Dict, List, Tuple
from .knowledge_base import GICSKnowledgeBase

class IndustryIntelligentAnalyzer:
    """
    Enhanced analyzer with industry intelligence
    """
    
    def __init__(self):
        self.knowledge_base = GICSKnowledgeBase()
        
    def analyze_document_structure(self, pdf_path: str) -> Dict:
        """
        Analyze document with industry intelligence
        """
        print("🧠 Industry-intelligent document analysis...")
        
        with pdfplumber.open(pdf_path) as pdf:
            # Sample pages for industry detection
            sample_text = ""
            company_name = ""
            
            # Extract text from first 10 pages for analysis
            for i in range(min(10, len(pdf.pages))):
                page_text = pdf.pages[i].extract_text() or ""
                sample_text += page_text + " "
                
                # Try to extract company name from early pages
                if i < 3 and not company_name:
                    potential_name = self._extract_company_name(page_text)
                    if potential_name != "Unknown Company":
                        company_name = potential_name
            
            # Final attempt at company name if not found
            if not company_name or company_name == "Unknown Company":
                company_name = self._extract_company_name(sample_text)
            
            # Detect industry
            industry_detection = self.knowledge_base.detect_industry(sample_text, company_name)
            
            # Analyze layout with industry context
            layout_analysis = self._analyze_layout_with_industry_context(pdf, industry_detection["industry"])
            
            print(f"  🏢 Company: {company_name}")
            print(f"  🏭 Industry: {industry_detection['industry']} (confidence: {industry_detection['confidence']:.2f})")
            print(f"  📊 Total pages: {len(pdf.pages)}")
            
            return {
                "company_name": company_name,
                "detected_industry": industry_detection,
                "industry_schema": self.knowledge_base.get_industry_info(industry_detection["industry"]),
                "layout_analysis": layout_analysis,
                "total_pages": len(pdf.pages),
                "sample_text": sample_text[:2000]  # Keep sample for further analysis
            }
    
    def _extract_company_name(self, text: str) -> str:
        """
        Extract company name from document text
        """
        # Enhanced company name patterns
        patterns = [
            # Standard corporate suffixes
            r'([A-Z][A-Za-z\s&\.]+(?:GROUP|PLC|INC|CORP|LIMITED|LTD|HOLDINGS|SA|AG|NV|LLC))',
            # Annual report patterns
            r'ANNUAL\s*REPORT\s*(?:2024|2025|2023|2022)\s*(?:FOR|OF)\s*([A-Z][A-Za-z\s,&\.]+)',
            # Financial report patterns
            r'([A-Z][A-Za-z\s&\.]+)\s*(?:ANNUAL|FINANCIAL)\s*REPORT',
            # Simple corporate patterns
            r'([A-Z][A-Za-z\s&\.]+)\s*(?:LIMITED|LTD|PLC|INC|CORP|HOLDINGS)',
            # Holdings patterns
            r'([A-Z][A-Za-z\s&\.]+)\s*HOLDINGS?\s*(?:PLC|INC|CORP|LIMITED|LTD)?',
            # Group patterns
            r'([A-Z][A-Za-z\s&\.]+)\s*GROUP\s*(?:PLC|INC|CORP|LIMITED|LTD)?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                company_name = match.strip()
                # Clean up the name
                company_name = re.sub(r'\s+', ' ', company_name)
                company_name = company_name.replace('\n', ' ')
                company_name = company_name.strip()
                
                # Validate the name (must be reasonable length and contain letters)
                if 3 <= len(company_name) <= 100 and re.search(r'[A-Za-z]', company_name):
                    # Remove common false positives
                    false_positives = [
                        'ANNUAL REPORT', 'FINANCIAL STATEMENTS', 'TABLE OF CONTENTS',
                        'CONSOLIDATED FINANCIAL', 'NOTES TO THE', 'INDEPENDENT AUDITOR',
                        'BOARD OF DIRECTORS', 'MANAGEMENT DISCUSSION'
                    ]
                    
                    if not any(fp in company_name.upper() for fp in false_positives):
                        return company_name
        
        return "Unknown Company"
    
    def _analyze_layout_with_industry_context(self, pdf, industry: str) -> Dict:
        """
        Analyze document layout with industry-specific priorities
        """
        layout_types = {}
        high_value_pages = []
        financial_pages = []
        
        # Industry-specific high-value sections
        industry_sections = {
            "airlines": [
                "operational statistics", "fleet information", "passenger data", 
                "route network", "load factor", "available seat", "aircraft"
            ],
            "banking": [
                "credit portfolio", "deposits", "branch network", "capital ratios",
                "loan portfolio", "interest margin", "basel", "tier 1"
            ],
            "technology": [
                "user metrics", "subscription data", "recurring revenue", "churn analysis",
                "active users", "customer acquisition", "saas metrics"
            ],
            "retail": [
                "store performance", "same store sales", "inventory data", "digital sales",
                "comparable sales", "store count", "e-commerce"
            ],
            "energy": [
                "production data", "reserves", "exploration results", "refining operations",
                "oil production", "gas reserves", "barrel", "drilling"
            ]
        }
        
        target_sections = industry_sections.get(industry, [])
        
        # Analyze sample of pages
        sample_size = min(20, len(pdf.pages))
        for i in range(sample_size):
            try:
                page = pdf.pages[i]
                text = page.extract_text() or ""
                
                # Classify layout
                layout_type = self._classify_page_layout(page, text, industry)
                layout_types[layout_type] = layout_types.get(layout_type, 0) + 1
                
                # Check for industry-specific high-value content
                text_lower = text.lower()
                
                # Industry-specific page scoring
                industry_score = 0
                for section in target_sections:
                    if section in text_lower:
                        industry_score += 1
                
                # Financial content scoring
                financial_keywords = [
                    'revenue', 'income', 'profit', 'assets', 'liabilities', 'cash flow',
                    'balance sheet', 'income statement', 'financial highlights'
                ]
                financial_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
                
                # Page priority scoring
                page_score = industry_score * 2 + financial_score
                
                if industry_score > 0:
                    high_value_pages.append({
                        "page": i + 1,
                        "score": page_score,
                        "type": "industry_specific"
                    })
                
                if financial_score > 2:
                    financial_pages.append({
                        "page": i + 1,
                        "score": financial_score,
                        "type": "financial"
                    })
                    
            except Exception as e:
                print(f"    ⚠️ Error analyzing page {i + 1}: {e}")
                continue
        
        # Sort pages by relevance
        high_value_pages.sort(key=lambda x: x["score"], reverse=True)
        financial_pages.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "layout_types": layout_types,
            "high_value_pages": [p["page"] for p in high_value_pages[:10]],
            "financial_pages": [p["page"] for p in financial_pages[:8]],
            "target_sections": target_sections,
            "page_scores": {
                "industry_specific": high_value_pages,
                "financial": financial_pages
            }
        }
    
    def _classify_page_layout(self, page, text: str, industry: str) -> str:
        """
        Classify page layout with industry context
        """
        # Check for tables first
        try:
            tables = page.extract_tables()
            if tables and len(tables) > 0:
                main_table = tables[0]
                
                # Industry-specific table detection
                text_lower = text.lower()
                
                if industry == "airlines":
                    airline_keywords = ["fleet", "aircraft", "passengers", "routes", "load factor"]
                    if any(keyword in text_lower for keyword in airline_keywords):
                        return "industry_specific_table"
                        
                elif industry == "banking":
                    banking_keywords = ["deposits", "loans", "branches", "capital", "interest"]
                    if any(keyword in text_lower for keyword in banking_keywords):
                        return "industry_specific_table"
                        
                elif industry == "technology":
                    tech_keywords = ["users", "revenue", "subscription", "churn", "arr"]
                    if any(keyword in text_lower for keyword in tech_keywords):
                        return "industry_specific_table"
                
                # General table classification
                if len(main_table) > 5 and len(main_table[0]) > 3:
                    return "structured_table"
                else:
                    return "simple_table"
                    
        except Exception as e:
            pass
        
        # Check for financial data patterns
        financial_patterns = [
            r'(?:total|net)\s+(?:assets|revenue|income)',
            r'\b\d{1,3}(?:,\d{3})*\s*(?:million|billion)',
            r'\b\d+\.\d+\s*%',
            r'(?:2024|2023|2022|2021|2020)'
        ]
        
        pattern_matches = sum(1 for pattern in financial_patterns 
                            if re.search(pattern, text, re.IGNORECASE))
        
        if pattern_matches >= 3:
            return "financial_data"
        
        # Check for narrative text
        sentences = text.split('.')
        if len(sentences) > 10 and len(text) > 1000:
            return "narrative_text"
        
        # Check for list/bullet format
        if re.search(r'^\s*[•\-\*]\s', text, re.MULTILINE):
            return "bullet_list"
        
        return "mixed_content"
    
    def create_industry_prompts(self, industry: str, page_analysis: Dict = None) -> Dict:
        """
        Create industry-specific extraction prompts
        """
        industry_info = self.knowledge_base.get_industry_info(industry)
        
        if not industry_info or not industry_info.get("key_metrics"):
            return {"primary": self._create_generic_prompt()}
        
        # Create comprehensive industry-specific prompt
        primary_prompt = self._create_industry_specific_prompt(industry, industry_info)
        
        # Create focused prompts for critical metrics
        focused_prompts = {}
        critical_metrics = self.knowledge_base.get_critical_metrics(industry)
        
        for metric in critical_metrics[:3]:  # Top 3 critical metrics
            if metric in industry_info["key_metrics"]:
                metric_info = industry_info["key_metrics"][metric]
                focused_prompts[f"focused_{metric}"] = self._create_metric_focused_prompt(
                    metric, metric_info
                )
        
        return {
            "primary": primary_prompt,
            "focused": focused_prompts,
            "fallback": self._create_generic_prompt()
        }
    
    def _create_industry_specific_prompt(self, industry: str, industry_info: Dict) -> str:
        """
        Create comprehensive industry-specific prompt
        """
        key_metrics = industry_info.get("key_metrics", {})
        business_questions = industry_info.get("business_questions", [])
        
        # Build metrics description
        metrics_description = []
        for metric, info in key_metrics.items():
            synonyms = ", ".join(info["synonyms"][:3])
            description = info.get("description", f"{metric} data")
            metrics_description.append(f"- {metric}: {description} (look for: {synonyms})")
        
        return f"""
INDUSTRY-SPECIFIC EXTRACTION FOR {industry.upper()}

You are analyzing a {industry} company's annual report. Extract metrics that matter for {industry} business analysis.

UNIVERSAL BUSINESS METRICS (extract if found):
- Total revenue/sales, net income, operating costs, operating income
- Employee count, total assets, total liabilities, cash flow from operations

{industry.upper()}-SPECIFIC METRICS (prioritize these):
{chr(10).join(metrics_description)}

KEY BUSINESS QUESTIONS TO ANSWER:
{chr(10).join(f"- {q}" for q in business_questions)}

EXTRACTION INSTRUCTIONS:
1. Extract ALL numerical values with their metric names
2. Identify time periods (FY2024, 2023, Q1 2024, etc.)
3. Determine units (millions, billions, percentage, count, aircraft, etc.)
4. Prioritize {industry}-specific operational metrics over general financial data
5. Look for growth rates, ratios, and efficiency metrics

REQUIRED JSON FORMAT:
{{"metric_name": "descriptive_name", "period": "time_period", "value": number, "unit": "unit_type"}}

CRITICAL: Return ONLY a valid JSON array. No explanations or other text.
"""
    
    def _create_metric_focused_prompt(self, metric: str, metric_info: Dict) -> str:
        """
        Create focused prompt for specific high-value metric
        """
        synonyms = ", ".join(metric_info["synonyms"])
        description = metric_info.get("description", f"{metric} data")
        unit = metric_info.get("unit", "unknown")
        
        return f"""
FOCUSED EXTRACTION: {metric.upper()}

TARGET METRIC: {metric}
DESCRIPTION: {description}
EXPECTED UNIT: {unit}

SEARCH TERMS: {synonyms}

INSTRUCTIONS:
1. Search specifically for this metric using the provided terms
2. Extract current year and previous years if available
3. Look for growth rates, changes, or trends related to this metric
4. Include any contextual data (quarterly breakdowns, regional splits, etc.)

REQUIRED JSON FORMAT:
{{"metric_name": "{metric}", "period": "time_period", "value": number, "unit": "{unit}"}}

Return ONLY the JSON array for this specific metric. If not found, return empty array [].
"""
    
    def _create_generic_prompt(self) -> str:
        """
        Create generic fallback prompt
        """
        return """
GENERAL FINANCIAL METRICS EXTRACTION

Extract all numerical financial and operational metrics from this text.

FOCUS ON:
- Revenue, income, costs, profits
- Employee data, customer data, operational statistics
- Growth rates, ratios, and performance metrics
- Assets, liabilities, cash flow data

REQUIRED JSON FORMAT:
{"metric_name": "descriptive_name", "period": "time_period", "value": number, "unit": "unit_type"}

Return ONLY a valid JSON array. No explanations.
"""
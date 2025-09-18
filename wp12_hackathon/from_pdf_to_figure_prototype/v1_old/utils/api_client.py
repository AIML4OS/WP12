"""
Unified LLM API client for metric extraction - Python 3.7+ compatible
"""

import requests
import json
import time
from typing import Dict, List, Optional

class LLMClient:
    """
    Unified client for LLM API interactions
    """
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Default timeouts by complexity
        self.timeouts = {
            "simple": 60,
            "complex": 120,
            "industry": 90
        }
    
    def extract_metrics(self, text: str, page_num: int, prompt: str, 
                       timeout: int = 90, context: str = "general") -> List[Dict]:
        """
        Extract metrics using LLM with standardized response parsing
        """
        try:
            # Create system message based on context
            system_message = self._create_system_message(context)
            
            # Prepare request
            data = {
                "model": "mistral-small3.2:latest",
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"{prompt}\n\nText:\n{text[:5000]}"}
                ],
                "temperature": 0.0,
                "max_tokens": 3000,
                "top_p": 0.1
            }
            
            # Make API call
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=data, 
                timeout=timeout
            )
            response.raise_for_status()
            
            # Parse response
            response_json = response.json()
            content = response_json['choices'][0]['message']['content'].strip()
            
            # Extract and parse JSON
            metrics = self._parse_metrics_response(content, page_num)
            
            return metrics
            
        except requests.exceptions.Timeout:
            print(f"    ⏰ LLM timeout for page {page_num}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"    ❌ API request failed for page {page_num}: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parsing failed for page {page_num}: {e}")
            return []
        except KeyError as e:
            print(f"    ❌ Response format error for page {page_num}: {e}")
            return []
        except Exception as e:
            print(f"    ❌ Metric extraction failed for page {page_num}: {e}")
            return []
    
    def _create_system_message(self, context: str) -> str:
        """Create appropriate system message based on context"""
        if context == "airlines":
            return "You are an expert airline industry analyst. Focus on operational metrics, fleet data, and passenger statistics."
        elif context == "banking":
            return "You are an expert banking analyst. Focus on deposits, loans, branches, and capital metrics."
        elif context == "technology":
            return "You are an expert technology analyst. Focus on user metrics, recurring revenue, and growth indicators."
        else:
            return "You are an expert financial analyst. Extract all relevant financial and operational metrics."
    
    def _parse_metrics_response(self, content: str, page_num: int) -> List[Dict]:
        """
        Parse LLM response and extract metrics in standardized format
        """
        try:
            # Find JSON array in response
            json_start = content.find('[')
            json_end = content.rfind(']')
            
            if json_start == -1 or json_end == -1:
                return []
            
            json_str = content[json_start:json_end + 1]
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                return []
            
            # Standardize metric format
            metrics = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                    
                # Extract required fields
                metric_name = item.get("metric_name", "")
                value = item.get("value", 0)
                
                if not metric_name or not isinstance(value, (int, float)):
                    continue
                
                # Create standardized metric
                metric = {
                    "metric": metric_name,
                    "value": float(value),
                    "unit": str(item.get("unit", "unknown")),
                    "period": str(item.get("period", "unknown")),
                    "confidence": 0.90,
                    "page_number": page_num,
                    "extraction_method": "llm_extraction",
                    "source_text": item.get("source_text", "")
                }
                
                metrics.append(metric)
            
            return metrics
            
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parsing failed for page {page_num}: {e}")
            return []
        except Exception as e:
            print(f"    ❌ Response parsing failed for page {page_num}: {e}")
            return []


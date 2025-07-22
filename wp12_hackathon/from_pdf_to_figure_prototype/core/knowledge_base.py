
# core/knowledge_base.py (Fixed version)
"""
Complete GICS-based industry knowledge base
"""

class GICSKnowledgeBase:
    """
    Complete industry knowledge base with proper initialization
    """
    
    def __init__(self):
        self.universal_metrics = self._initialize_universal_metrics()
        self.industry_schemas = self._initialize_industry_schemas()
    
    def _initialize_universal_metrics(self) -> dict:
        """Initialize universal business metrics"""
        return {
            "total_revenue": {
                "synonyms": ["total sales", "net sales", "turnover", "revenue", "total revenue"],
                "unit": "millions_eur",
                "description": "Total company revenue"
            },
            "operating_costs": {
                "synonyms": ["operating expenses", "cost of goods sold", "cogs", "operating costs"],
                "unit": "millions_eur",
                "description": "Total operating expenses"
            },
            "net_income": {
                "synonyms": ["profit for the year", "profit after tax", "earnings", "net income"],
                "unit": "millions_eur",
                "description": "Company profit after taxes"
            },
            "employee_count": {
                "synonyms": ["number of employees", "headcount", "ftes", "workforce"],
                "unit": "count",
                "description": "Total number of employees"
            },
            "total_assets": {
                "synonyms": ["total assets", "assets"],
                "unit": "millions_eur",
                "description": "Total company assets"
            },
            "cash_flow_from_operations": {
                "synonyms": ["operating cash flow", "cash flow from operations"],
                "unit": "millions_eur",
                "description": "Cash generated from operations"
            }
        }
    
    def _initialize_industry_schemas(self) -> dict:
        """Initialize industry-specific schemas"""
        return {
            "airlines": {
                "display_name": "Airlines",
                "sector": "Industrials",
                "key_metrics": {
                    "fleet_size": {
                        "synonyms": ["number of aircraft", "fleet size", "aircraft fleet", "fleet"],
                        "unit": "aircraft",
                        "importance": "high",
                        "description": "Total number of aircraft in operation"
                    },
                    "passengers_carried": {
                        "synonyms": ["passengers carried", "passenger numbers", "passenger traffic", "passengers"],
                        "unit": "millions",
                        "importance": "high",
                        "description": "Total passengers transported annually"
                    },
                    "load_factor": {
                        "synonyms": ["load factor", "passenger load factor", "seat load factor"],
                        "unit": "percentage",
                        "importance": "critical",
                        "description": "Percentage of available seats filled"
                    },
                    "available_seat_km": {
                        "synonyms": ["available seat kilometres", "ask", "seat capacity"],
                        "unit": "millions",
                        "importance": "medium",
                        "description": "Total passenger capacity offered"
                    }
                },
                "business_questions": [
                    "How efficiently do they operate their fleet?",
                    "Are their planes full (load factor)?",
                    "What's their cost structure per passenger?",
                    "How does their route network perform?"
                ]
            },
            "banking": {
                "display_name": "Banking",
                "sector": "Financials",
                "key_metrics": {
                    "net_interest_margin": {
                        "synonyms": ["net interest margin", "nim", "interest margin"],
                        "unit": "percentage",
                        "importance": "critical",
                        "description": "Profitability of lending operations"
                    },
                    "number_of_branches": {
                        "synonyms": ["number of branches", "branch network", "agencies", "branches"],
                        "unit": "count",
                        "importance": "high",
                        "description": "Physical presence and reach"
                    },
                    "deposits": {
                        "synonyms": ["customer deposits", "total deposits", "deposit base", "deposits"],
                        "unit": "millions_eur",
                        "importance": "high",
                        "description": "Customer funds held by the bank"
                    },
                    "loan_portfolio": {
                        "synonyms": ["loan portfolio", "total loans", "advances", "loans"],
                        "unit": "millions_eur",
                        "importance": "high",
                        "description": "Total loans outstanding"
                    },
                    "tier_1_capital": {
                        "synonyms": ["tier 1 capital", "tier 1 ratio", "capital ratio"],
                        "unit": "percentage",
                        "importance": "critical",
                        "description": "Bank's core capital adequacy"
                    }
                },
                "business_questions": [
                    "How profitable is their lending business?",
                    "What's their market presence?",
                    "How well capitalized are they?",
                    "What's their deposit growth?"
                ]
            },
            "technology": {
                "display_name": "Technology",
                "sector": "Information Technology",
                "key_metrics": {
                    "annual_recurring_revenue": {
                        "synonyms": ["annual recurring revenue", "arr", "recurring revenue"],
                        "unit": "millions_eur",
                        "importance": "critical",
                        "description": "Predictable yearly subscription revenue"
                    },
                    "active_users": {
                        "synonyms": ["active users", "monthly active users", "user base", "users"],
                        "unit": "millions",
                        "importance": "high",
                        "description": "Number of engaged users"
                    },
                    "churn_rate": {
                        "synonyms": ["churn rate", "customer churn", "attrition", "churn"],
                        "unit": "percentage",
                        "importance": "critical",
                        "description": "Rate of customer loss"
                    },
                    "customer_acquisition_cost": {
                        "synonyms": ["customer acquisition cost", "cac", "acquisition cost"],
                        "unit": "eur",
                        "importance": "medium",
                        "description": "Cost to acquire new customers"
                    }
                },
                "business_questions": [
                    "Is their recurring revenue growing?",
                    "Are they retaining customers?",
                    "What's their growth trajectory?",
                    "How efficient is their customer acquisition?"
                ]
            },
            "retail": {
                "display_name": "Retail",
                "sector": "Consumer Discretionary",
                "key_metrics": {
                    "same_store_sales": {
                        "synonyms": ["same store sales", "comparable store sales", "like for like"],
                        "unit": "percentage",
                        "importance": "critical",
                        "description": "Sales growth in existing stores"
                    },
                    "number_of_stores": {
                        "synonyms": ["number of stores", "store count", "outlets", "stores"],
                        "unit": "count",
                        "importance": "high",
                        "description": "Total retail locations"
                    },
                    "inventory_turnover": {
                        "synonyms": ["inventory turnover", "stock turnover", "inventory turns"],
                        "unit": "ratio",
                        "importance": "medium",
                        "description": "Efficiency of inventory management"
                    },
                    "sales_per_square_meter": {
                        "synonyms": ["sales per square meter", "sales density", "productivity"],
                        "unit": "eur_per_sqm",
                        "importance": "medium",
                        "description": "Revenue efficiency per store area"
                    }
                },
                "business_questions": [
                    "How are existing stores performing?",
                    "What's their expansion strategy?",
                    "How efficiently do they manage inventory?",
                    "What's their store productivity?"
                ]
            }
        }
    
    def detect_industry(self, text: str, company_name: str = "") -> dict:
        """
        Detect industry with improved confidence scoring
        """
        text_lower = text.lower()
        company_lower = company_name.lower()
        
        # Enhanced industry detection patterns
        industry_indicators = {
            "airlines": [
                "aircraft", "flights", "passengers", "aviation", "airline", "fleet",
                "load factor", "available seat", "airports", "routes", "cabin crew",
                "boeing", "airbus", "runway", "departure", "arrival"
            ],
            "banking": [
                "deposits", "loans", "branches", "bank", "credit", "capital",
                "tier 1", "basel", "regulatory capital", "net interest margin",
                "mortgages", "savings", "current account", "atm"
            ],
            "technology": [
                "software", "saas", "users", "platform", "digital", "cloud",
                "subscription", "app", "api", "data", "analytics", "artificial intelligence"
            ],
            "retail": [
                "stores", "retail", "same store sales", "inventory", "merchandise",
                "outlets", "shopping", "consumer", "customers", "sales floor",
                "e-commerce", "online", "mall", "supermarket"
            ],
            "energy": [
                "oil", "gas", "petroleum", "barrels", "reserves", "exploration",
                "production", "refining", "energy", "upstream", "downstream",
                "crude", "natural gas", "drilling", "pipeline"
            ]
        }
        
        # Calculate scores for each industry
        industry_scores = {}
        for industry, keywords in industry_indicators.items():
            text_score = sum(text_lower.count(keyword) for keyword in keywords)
            company_score = sum(company_lower.count(keyword) * 3 for keyword in keywords)  # Company name weighted higher
            
            # Additional scoring for industry-specific phrases
            phrase_score = 0
            if industry == "airlines":
                phrases = ["passenger load factor", "available seat kilometres", "fleet utilization"]
                phrase_score = sum(2 for phrase in phrases if phrase in text_lower)
            elif industry == "banking":
                phrases = ["net interest margin", "loan loss provisions", "capital adequacy"]
                phrase_score = sum(2 for phrase in phrases if phrase in text_lower)
            
            total_score = text_score + company_score + phrase_score
            industry_scores[industry] = total_score
        
        # Determine best match
        if not industry_scores or max(industry_scores.values()) < 3:
            return {"industry": "other", "confidence": 0.0}
        
        detected_industry = max(industry_scores, key=industry_scores.get)
        max_score = industry_scores[detected_industry]
        
        # Calculate confidence (0.0 to 1.0)
        confidence = min(max_score / 20, 1.0)  # Normalize score
        
        return {
            "industry": detected_industry,
            "confidence": confidence,
            "scores": industry_scores
        }
    
    def get_industry_info(self, industry: str) -> dict:
        """Get comprehensive industry information"""
        if industry not in self.industry_schemas:
            return {
                "key_metrics": {},
                "business_questions": [],
                "display_name": "Other",
                "sector": "Unknown"
            }
        
        return self.industry_schemas[industry]
    
    def get_critical_metrics(self, industry: str) -> list[str]:
        """Get the most important metrics for an industry"""
        if industry not in self.industry_schemas:
            return list(self.universal_metrics.keys())
        
        industry_info = self.industry_schemas[industry]
        metrics = industry_info["key_metrics"]
        
        # Get critical metrics first
        critical = [name for name, info in metrics.items() 
                   if info["importance"] == "critical"]
        
        # Add high importance metrics
        high = [name for name, info in metrics.items() 
                if info["importance"] == "high"]
        
        # Add some universal metrics
        universal = list(self.universal_metrics.keys())[:3]
        
        return critical + high + universal
    
    def get_all_target_metrics(self, industry: str) -> list[str]:
        """Get all metrics to search for (universal + industry-specific)"""
        universal_metrics = list(self.universal_metrics.keys())
        
        if industry in self.industry_schemas:
            industry_metrics = list(self.industry_schemas[industry]["key_metrics"].keys())
            return universal_metrics + industry_metrics
        
        return universal_metrics


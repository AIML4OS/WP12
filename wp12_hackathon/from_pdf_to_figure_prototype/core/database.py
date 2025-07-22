# core/unified_database.py
"""
Unified database manager resolving all inconsistencies
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class DatabaseManager:
    """
    Unified database manager that resolves all schema inconsistencies
    """
    
    def __init__(self, db_path: str = "intelligent_rag.db"):
        self.db_path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row  # Enable column access by name
        self._setup_unified_schema()
    
    def _setup_unified_schema(self):
        """Create unified schema that resolves all inconsistencies"""
        cursor = self.connection.cursor()
        
        # Companies table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                sector TEXT,
                detected_industry TEXT,
                industry_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Documents table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id INTEGER,
                filename TEXT NOT NULL,
                file_path TEXT,
                total_pages INTEGER,
                pages_analyzed INTEGER,
                pages_processed INTEGER,
                processing_time REAL,
                processing_strategy TEXT,
                status TEXT DEFAULT 'processing',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES companies (id)
            )
        """)
        
        # Industry analysis table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS industry_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                detected_industry TEXT NOT NULL,
                industry_confidence REAL,
                detection_scores TEXT,  -- JSON object with all industry scores
                target_metrics TEXT,    -- JSON array of target metrics
                layout_analysis TEXT,   -- JSON object of layout analysis
                extraction_strategy TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Financial metrics table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                page_number INTEGER,
                metric_name TEXT NOT NULL,
                metric_type TEXT,  -- 'universal', 'industry_specific', 'other'
                value REAL NOT NULL,
                unit TEXT,
                period TEXT,
                confidence REAL,
                extraction_method TEXT,
                source_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Business intelligence table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS business_intelligence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                concept TEXT NOT NULL,
                insight_text TEXT NOT NULL,
                supporting_metrics TEXT,  -- JSON array of metric names
                confidence REAL,
                insight_type TEXT DEFAULT 'general',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Page metadata table - unified version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS page_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                page_number INTEGER,
                page_type TEXT,
                priority_score REAL,
                industry_relevance_score REAL,
                contains_tables BOOLEAN,
                contains_financial_data BOOLEAN,
                processed BOOLEAN DEFAULT FALSE,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Processing logs table - new for tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER,
                stage TEXT,
                message TEXT,
                level TEXT DEFAULT 'INFO',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents (id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_document ON financial_metrics(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON financial_metrics(metric_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_document ON business_intelligence(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_companies_industry ON companies(detected_industry)")
        
        self.connection.commit()
        print("✅ Unified database schema created successfully")
    
    def get_connection(self):
        """Get database connection for direct access"""
        return self.connection
    
    def close(self):
        """Close database connection"""
        self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
import os
from pathlib import Path

class Config:
    """Unified configuration class with validation"""
    
    # API Configuration
    API_KEY = os.getenv('API_KEY', 'sk-47e33d0d83f64299949c78b961956144')
    BASE_URL = os.getenv('BASE_URL', 'https://llm.lab.sspcloud.fr/api/chat/completions')
    
    # File Upload Configuration
    UPLOAD_FOLDER = os.path.join(Path(__file__).parent, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # Database Configuration
    DATABASE_PATH = os.path.join(Path(__file__).parent, 'intelligent_rag.db')
    
    # Processing Configuration
    MAX_PROCESSING_TIME = 900  # 15 minutes
    MAX_PAGES_TO_PROCESS = 20
    PROCESSING_BATCH_SIZE = 4
    
    # Security
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here-change-in-production')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.path.join(Path(__file__).parent, 'app.log')
    
    # Performance
    REQUEST_TIMEOUT = 30
    LLM_TIMEOUT = 90
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if not cls.API_KEY or cls.API_KEY == 'your-api-key-here':
            errors.append("API_KEY must be set")
        
        if not cls.BASE_URL:
            errors.append("BASE_URL must be set")
        
        if cls.SECRET_KEY == 'your-secret-key-here-change-in-production':
            errors.append("SECRET_KEY should be changed in production")
        
        if errors:
            raise ValueError("Configuration errors: " + ", ".join(errors))
        
        return True
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Path(self.DATABASE_PATH).parent, exist_ok=True)
        os.makedirs(Path(self.LOG_FILE).parent, exist_ok=True)

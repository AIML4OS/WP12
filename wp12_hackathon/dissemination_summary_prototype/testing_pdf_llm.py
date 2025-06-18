import uvicorn
from fastapi_main import app
import threading
import subprocess
from pathlib import Path


SYSTEM_URL = "http://127.0.0.1:8000/pdf_summarizer"
    
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8000)
# starting fastapi as non-blocking thread
fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()
# launching streamlit app
dir = Path(__file__).parent
streamlit_path = dir/"streamlit_main.py"
subprocess.run(["streamlit", "run", streamlit_path])


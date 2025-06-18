from fastapi import FastAPI
from ollama_pdf import review_chain
from frontend_models import SummaryRequestInput,SummaryRequestOutput
import uvicorn


app = FastAPI(
    title="PDF Summarizer",
    description="Endpoints for a AI-based pdf summarizer system ",
)

@app.get("/")
async def get_status():
    return {"status": "running"}

async def invoke_system(summary_request: str) -> SummaryRequestOutput:
    """
    Invoke the system to return a summary of the pdf.
    """
    summary = await review_chain.ainvoke(summary_request)
    return SummaryRequestOutput(output=summary)


@app.post("/pdf_summarizer")
async def query_hospital_agent(query: SummaryRequestInput) -> SummaryRequestOutput:
    query_response = await invoke_system(query.text)
    return query_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
import os
import requests
import streamlit as st
from frontend_models import SummaryRequestInput


SYSTEM_URL = "http://127.0.0.1:8000/pdf_summarizer"

st.title("Simple RAG Application")
st.write("Choose a Customized Summary")
if st.button("Short English Airflights"):
    try:
        # Make sure the FastAPI server is running before making a request
        # A short delay can help ensure the server is up
        text="Do a short summary of the Airflights statistics in Portugal"
        with st.spinner("Summarizing..."):
            response = requests.post(SYSTEM_URL, json={"text":text})
            response.raise_for_status()  # Raise an exception for bad status codes
            answer = response.content
            st.write("### Answer")
            st.write(answer)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
        st.info("Please make sure the backend server is running.")

st.subheader("📄 PDF Uploader")

# 1) File uploader that only accepts PDFs
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
if uploaded_file:
    # 2) Prepare the file for multipart/form-data
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
    }
    # 3) POST to the FastAPI endpoint
    with st.spinner("Uploading…"):
        resp = requests.post("http://localhost:8000/upload-pdf/", files=files)
    if resp.ok:
        data = resp.json()
        st.success(f"Uploaded **{data['filename']}** ({data['size_bytes']} bytes)")
    else:
        st.error(f"Upload failed: {resp.status_code}")


#!/bin/bash

git clone https://github.com/AIML4OS/WP12.git
cd WP12/wp12_hackathon/dissemination_summary_prototype/prototype_b/
python3 -m venv .venv_pdf
source .venv_pdf/bin/activate
pip install -r requirements.txt

# Ollama install
curl -fsSL https://ollama.com/install.sh | sh
sleep 5
nohup ollama serve &
sleep 5
echo "/bye" | ollama run llama3.2 # pulling the llama3.2 model

# launching App
source .venv_pdf/bin/activate
python main.py
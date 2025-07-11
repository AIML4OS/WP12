#!/bin/bash

echo "Starting my app and health server..."

# Start a simple Python HTTP server in the background for health checks
# This serves a file named 'healthz' at the root.
# Ensure 'healthz' file exists and contains "OK" for healthy state.
echo "OK" > /tmp/healthz
(cd /tmp && python3 -m http.server 8080 &)
HTTP_SERVER_PID=$!
echo "Health server PID: $HTTP_SERVER_PID"
git clone https://github.com/AIML4OS/WP12.git
cd WP12/wp12_hackathon/dissemination_summary_prototype/prototype_b/
python3 -m venv .venv_pdf
source .venv_pdf/bin/activate
nohup pip install -r requirements.txt &
# Ollama install
nohup curl -fsSL https://ollama.com/install.sh | sh &
sleep 5
nohup ollama serve &
sleep 5
ollama pull llama3.2 # pulling the llama3.2 model
# launching App
source .venv_pdf/bin/activate
python main.py






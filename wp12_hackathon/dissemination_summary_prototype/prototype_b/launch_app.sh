./set_ollama.sh &
OLLAMA_PID=$!

./set_venv.sh

if ps -p $OLLAMA_PID > /dev/null; then
    echo "Ollama script is still running"
    wait OLLAMA_PID

# launching App
source .venv_pdf/bin/activate
python main.py
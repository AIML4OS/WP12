# Ollama install
curl -fsSL https://ollama.com/install.sh | sh
sleep 5
nohup ollama serve & # serve blocks the terminal
sleep 5
ollama pull llama3.2 # pulling the llama3.2 model
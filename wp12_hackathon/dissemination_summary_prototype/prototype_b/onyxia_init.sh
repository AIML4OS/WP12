git clone https://github.com/AIML4OS/WP12.git
cd WP12/wp12_hackathon/dissemination_summary_prototype/prototype_b/
python3 -m venv .venv_pdf
source .venv_pdf/bin/activate
pip install -r requirements.txt

# Ollama install
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve &
echo "/bye" | ollama run llama3.2 # pulling the llama3.2 model and exiting it.

# launching App
source WP12/wp12_hackathon/dissemination_summary_prototype/prototype_b/.venv_pdf/bin/activate
python WP12/wp12_hackathon/dissemination_summary_prototype/prototype_b/main.py
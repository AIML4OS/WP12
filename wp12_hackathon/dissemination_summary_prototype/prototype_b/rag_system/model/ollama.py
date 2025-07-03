from langchain_ollama import OllamaLLM
# Setting the LLM

# doesn't nothing special, but we may need to extend it later on...
class Ollama:
    @property
    def core(self,model="llama3.2", temperature=0, top_p=0.9, num_predict=100-150, **kwargs):
        return OllamaLLM(
            model=model, 
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict, 
            **kwargs
        )
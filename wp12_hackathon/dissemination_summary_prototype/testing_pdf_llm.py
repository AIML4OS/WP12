from ollama_pdf import review_chain

question = """
Qual a recente tendência no movimento de passageiros nos aeroportos nacionais? 
"""
print(review_chain.invoke(question))


question = """
What can you tell me about vital statistics in Portugal? 
"""
print(review_chain.invoke(question))
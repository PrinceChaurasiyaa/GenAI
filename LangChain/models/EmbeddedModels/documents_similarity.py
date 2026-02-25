from langchain_ollama import OllamaEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = OllamaEmbeddings(
    model="mxbai-embed-large"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

prompt = "Who is famous for God of Cricket"

doc_embeddings = embedding.embed_documents(documents)
prompt_embedding = embedding.embed_query(prompt)

score = cosine_similarity([prompt_embedding], doc_embeddings)[0]
indexed_score = list(enumerate(score))
print(indexed_score)

index, sorted_score = sorted(indexed_score, key=lambda x:x[1])[-1]
print(prompt)
print(documents[index])
print("similarity score is:", sorted_score)

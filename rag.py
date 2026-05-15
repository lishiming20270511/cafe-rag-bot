import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic


def load_faqs(filepath):
    import re
    faqs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    q = ''
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        q_match = re.match(r'^Q\d*[:.]\s*(.+)', line)
        a_match = re.match(r'^A\d*[:.]\s*(.+)', line)
        if q_match:
            q = q_match.group(1)
        elif a_match and q:
            a = a_match.group(1)
            faqs.append({'question': q, 'answer': a, 'text': f"Q: {q}\nA: {a}"})
            q = ''
    return faqs


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class RAGBot:
    def __init__(self, faq_path='cafe_faq.txt'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.faqs = load_faqs(faq_path)
        self.embeddings = self.model.encode([faq['text'] for faq in self.faqs])
        self.client = anthropic.Anthropic(
            base_url='https://api.pateway.ai',
            api_key=os.environ.get('PATEWAY_API_KEY')
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query])[0]
        similarities = [cosine_similarity(query_embedding, emb) for emb in self.embeddings]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.faqs[i] for i in top_indices]

    def generate(self, query, retrieved_faqs):
        context = '\n\n'.join([faq['text'] for faq in retrieved_faqs])
        response = self.client.messages.create(
            model='claude-haiku-4-5-20251001',
            max_tokens=1024,
            system="You are a helpful cafe assistant. Answer questions based on the provided FAQ context. If the answer isn't in the context, say you don't have that information.",
            messages=[{
                'role': 'user',
                'content': f"Based on the following FAQ information:\n\n{context}\n\nPlease answer this question: {query}"
            }]
        )
        return response.content[0].text

    def answer(self, query):
        retrieved = self.retrieve(query)
        return self.generate(query, retrieved)

import os
import re
import numpy as np
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

load_dotenv()


def load_faqs(filepath):
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


class RAGBot:
    def __init__(self, faq_path='cafe_faq.txt'):
        self.faqs = load_faqs(faq_path)
        texts = [faq['text'] for faq in self.faqs]
        self.vectorizer = TfidfVectorizer()
        self.embeddings = self.vectorizer.fit_transform(texts)
        self.client = anthropic.Anthropic(
            base_url='https://api.pateway.ai',
            api_key=os.environ.get('PATEWAY_API_KEY')
        )

    def retrieve(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
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

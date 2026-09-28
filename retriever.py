"""
CSC-128 Assignment 6: TF-IDF retrieval over the knowledge base
Archit Dubey

Imports from knowledge.py only. Nothing here imports Streamlit or groq, so the
tests run with no API key at all.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge import CHUNKS

# Threshold set from the scores printed by test_retriever.py, not guessed.
# Lowest score among should-retrieve: 0.201. Highest among should-refuse: 0.000.
# 0.10 sits in the middle of that gap. See the README.
DEFAULT_THRESHOLD = 0.10


class Retriever:
    def __init__(self, chunks=None, threshold=DEFAULT_THRESHOLD):
        if chunks is None:
            chunks = CHUNKS

        self.chunks = chunks
        self.threshold = threshold

        texts = [chunk["text"] for chunk in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, question, top_k=3):
        """Return a list of (chunk, score) above the threshold, best first."""
        vector = self.vectorizer.transform([question])
        scores = cosine_similarity(vector, self.matrix)[0]

        results = []
        for chunk, score in zip(self.chunks, scores):
            if score >= self.threshold:
                results.append((chunk, float(score)))

        results.sort(key=lambda pair: pair[1], reverse=True)
        return results[:top_k]

    def best_score(self, question):
        """Return the single highest score for a question, ignoring the threshold."""
        vector = self.vectorizer.transform([question])
        scores = cosine_similarity(vector, self.matrix)[0]
        return float(max(scores))
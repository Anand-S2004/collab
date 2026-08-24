import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_chunks(question, docs, k=8, expand=True):
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(d["text"] for d in docs)

    def score(query):
        sims = cosine_similarity(vectorizer.transform([query]), doc_vectors)[0]
        ranked = np.argsort(-sims, kind="stable")
        return [i for i in ranked if sims[i] > 0][:k]

    hits = score(question)
    if expand:
        query = question + " " + " ".join(docs[i]["text"] for i in hits)
        hits = score(query)
    return [docs[i] for i in hits]


def all_chunks(question, docs, k=None, expand=None):
    return list(docs)

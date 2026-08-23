"""Text-retrieval arm of the KG-vs-text benchmark.

Chunk -> rank by TF-IDF cosine similarity -> hand top-k to a local model -> print.
Data is hardcoded until the generator's output format is settled.
"""

import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3.2"

DOCS = [
    {"id": "d1", "text": "Alice Chen is a senior engineer at Acme Corp."},
    {"id": "d2", "text": "Acme Corp was acquired by Globex Industries in 2019."},
    {"id": "d3", "text": "Globex Industries is headquartered in Zurich."},
    {"id": "d4", "text": "Bob Reyes manages the platform team at Initech."},
    {"id": "d5", "text": "Initech uses PostgreSQL for all production databases."},
]

QUESTIONS = [
    {"q": "Where does Alice Chen work?", "answer": "Acme Corp"},
    {"q": "Who ultimately owns the company Alice Chen works for?", "answer": "Globex Industries"},
    {"q": "What database does Bob Reyes's company use?", "answer": "PostgreSQL"},
]

PROMPT = """Answer the question using only the context below.

Context:
{context}

Question: {question}

Answer exactly what was asked. Give the most direct answer the context states - do
not substitute a parent, owner, or related entity unless the question asks for one.
Reply with just the answer and nothing else - no explanation, no full sentence.
If the context does not contain the answer, reply exactly: NOT IN CONTEXT"""


def rank_chunks(question, docs, k=3):
    """Return up to k docs relevant to the question, by TF-IDF cosine similarity.

    Two passes. The first scores docs against the question alone, which only ever
    reaches docs that share vocabulary with it. Multi-hop questions don't: "who owns
    Alice Chen's company" has no lexical overlap with the doc naming the acquirer,
    so pass two re-scores against the question plus the text of whatever pass one
    found - once "Acme Corp" is in the query, the acquisition doc becomes reachable.

    Only docs with nonzero similarity are returned, so a thin result stays thin
    rather than being padded with arbitrary zero-scoring docs.

    One chunk per doc - they're a sentence each. Swapping TF-IDF for embeddings
    later means rewriting this function and nothing else.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(d["text"] for d in docs)

    def score(query):
        sims = cosine_similarity(vectorizer.transform([query]), doc_vectors)[0]
        ranked = np.argsort(-sims, kind="stable")
        return [i for i in ranked if sims[i] > 0][:k]

    seeds = score(question)
    expanded = question + " " + " ".join(docs[i]["text"] for i in seeds)
    return [docs[i] for i in score(expanded)]


def ask_model(question, context_chunks):
    """Ask the local model the question against the retrieved chunks."""
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in context_chunks)
    prompt = PROMPT.format(context=context, question=question)

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        # A benchmark wants the same answer every run - otherwise arms differ by noise.
        options={"temperature": 0, "seed": 0},
    )
    return resp["message"]["content"].strip()


def main():
    for item in QUESTIONS:
        chunks = rank_chunks(item["q"], DOCS)
        answer = ask_model(item["q"], chunks)

        print(f"Q:        {item['q']}")
        print(f"retrieved {[c['id'] for c in chunks]}")
        print(f"model:    {answer}")
        print(f"expected: {item['answer']}")
        print()


if __name__ == "__main__":
    main()

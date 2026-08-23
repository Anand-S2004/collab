"""Retrieval arms of the KG-vs-text benchmark.

Two arms tonight, sharing one model and one prompt so they differ only by
retrieval:
  text          - rank chunks by TF-IDF cosine similarity, pass the top k
  full-context  - no retrieval, pass every doc

Data is hardcoded until the generator's output format is settled.
"""

import argparse

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


def rank_chunks(question, docs, k=3, expand=True):
    """Return up to k docs relevant to the question, by TF-IDF cosine similarity.

    A single pass only reaches docs sharing vocabulary with the question, which
    multi-hop questions don't: "who owns Alice Chen's company" has no lexical
    overlap with the doc naming the acquirer. With expand=True a second pass
    re-scores against the question plus whatever the first pass found - once
    "Acme Corp" is in the query, the acquisition doc becomes reachable.

    expand=False is the weaker single-pass baseline, kept so the two can be
    compared directly. On this corpus it cannot answer the multi-hop questions.

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

    hits = score(question)
    if expand:
        query = question + " " + " ".join(docs[i]["text"] for i in hits)
        hits = score(query)
    return [docs[i] for i in hits]


def all_chunks(question, docs):
    """Full-context arm: no retrieval, every doc goes to the model.

    Same signature as rank_chunks so the two are interchangeable.

    This is the control, but it is NOT an upper bound - it already loses to the
    text arm on Q3 at five documents. Worse, its answer changes with the order of
    DOCS: [d4, d5] answers Q3 correctly and [d3, d4, d5] does not, on identical
    relevant content. Whatever this arm scores, some of it is doc ordering rather
    than the model's ability to use the context. Treat it as a reference point,
    and hold the ordering fixed when comparing arms.
    """
    return list(docs)


def ask_model(question, context_chunks):
    """Ask the local model the question against the given chunks."""
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in context_chunks)
    prompt = PROMPT.format(context=context, question=question)

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        # A benchmark wants the same answer every run - otherwise arms differ by noise.
        options={"temperature": 0, "seed": 0},
    )
    return resp["message"]["content"].strip()


def normalize(text):
    """Fold the differences that shouldn't count as a wrong answer."""
    return text.strip().lower().rstrip(".").strip()


def format_ids(chunks, limit=4):
    """Doc ids, abbreviated - the full-context arm gets long as the corpus grows."""
    ids = [c["id"] for c in chunks]
    if len(ids) > limit:
        return f"{', '.join(ids[:limit])} +{len(ids) - limit} more"
    return ", ".join(ids) if ids else "(nothing retrieved)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="disable the text arm's second retrieval pass (query expansion)",
    )
    args = parser.parse_args()

    arms = [
        (f"text{'' if args.expand else ' (no expand)'}",
         lambda q, docs: rank_chunks(q, docs, expand=args.expand)),
        ("full-context", all_chunks),
    ]

    for item in QUESTIONS:
        print(f"Q: {item['q']}")
        print(f"   expected: {item['answer']}")
        for name, retrieve in arms:
            chunks = retrieve(item["q"], DOCS)
            answer = ask_model(item["q"], chunks)
            hit = "PASS" if normalize(answer) == normalize(item["answer"]) else "FAIL"
            print(f"   {name:<18} {hit}  {answer!r}")
            print(f"   {'':<18}       [{format_ids(chunks)}]")
        print()


if __name__ == "__main__":
    main()

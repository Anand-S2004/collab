import argparse
import json

import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3.2"

PROMPT = """You are answering from a list of facts. Each line is tagged [dN]; the tag is a
label, never part of the answer.

Facts:
{context}

Question: {question}

The facts may need to be combined: one fact may name a thing, and another fact may
say something about that thing. Work through them before answering.
Reply with only the answer - a name or a short noun phrase, nothing else.
Only if no combination of facts answers it, reply exactly: NOT IN CONTEXT"""

ARTICLES = ("the ", "a ", "an ")


def load_docs(path):
    with open(path) as fh:
        return json.load(fh)


def load_questions(path):
    with open(path) as fh:
        return json.load(fh)


def rank_chunks(question, docs, k=3, expand=True):
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


def ask_model(question, context_chunks):
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in context_chunks)
    prompt = PROMPT.format(context=context, question=question)

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "seed": 0},
    )
    return resp["message"]["content"].strip()


def normalize(text):
    text = text.strip().lower().rstrip(".").strip()
    for article in ARTICLES:
        if text.startswith(article):
            return text[len(article):]
    return text


def format_ids(chunks, limit=6):
    ids = [c["id"] for c in chunks]
    if len(ids) > limit:
        return f"{', '.join(ids[:limit])} +{len(ids) - limit} more"
    return ", ".join(ids) if ids else "(nothing retrieved)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/docs.json")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--no-expand", dest="expand", action="store_false")
    args = parser.parse_args()

    docs = load_docs(args.docs)
    questions = load_questions(args.questions)
    print(f"{len(docs)} docs from {args.docs}, {len(questions)} questions from {args.questions}\n")

    arms = [
        (f"text{'' if args.expand else ' (no expand)'}", rank_chunks),
        ("full-context", all_chunks),
    ]
    tally = {name: 0 for name, _ in arms}

    for item in questions:
        print(f"{item['id']} ({item['hops']}-hop): {item['q']}")
        print(f"   expected: {item['answer']}   supported by: {', '.join(item['docs'])}")
        for name, retrieve in arms:
            chunks = retrieve(item["q"], docs, k=args.k, expand=args.expand)
            answer = ask_model(item["q"], chunks)

            correct = normalize(answer) == normalize(item["answer"])
            tally[name] += correct
            missing = set(item["docs"]) - {c["id"] for c in chunks}
            retrieval = "got support" if not missing else f"MISSING {','.join(sorted(missing))}"

            print(f"   {name:<18} {'PASS' if correct else 'FAIL'}  {answer!r}")
            print(f"   {'':<18} {retrieval} [{format_ids(chunks)}]")
        print()

    print("=" * 60)
    for name, _ in arms:
        print(f"{name:<18} {tally[name]}/{len(questions)}")


if __name__ == "__main__":
    main()

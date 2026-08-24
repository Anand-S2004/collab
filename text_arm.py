"""Retrieval arms of the KG-vs-text benchmark.

Two arms, sharing one model and one prompt so they differ only by retrieval:
  text          - rank chunks by TF-IDF cosine similarity, pass the top k
  full-context  - no retrieval, pass every doc

Documents come from the generator (data/docs.json). Questions are hand-written
against a specific generator seed and live in data/questions.json, so the graph
arm can run the identical quiz - the comparison is only meaningful if both arms
answer the same questions with the same model and the same prompt.
"""

import argparse
import json

import numpy as np
import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "llama3.2"

# Both arms share this. The graph arm has to use it too, or the benchmark measures
# prompt wording instead of retrieval. Tagging matters: an earlier version omitted
# "the tag is a label" and the model answered a question with "[d22]".
PROMPT = """You are answering from a list of facts. Each line is tagged [dN]; the tag is a
label, never part of the answer.

Facts:
{context}

Question: {question}

The facts may need to be combined: one fact may name a thing, and another fact may
say something about that thing. Work through them before answering.
Reply with only the answer - a name or a short noun phrase, nothing else.
Only if no combination of facts answers it, reply exactly: NOT IN CONTEXT"""


def load_docs(path):
    """Read the generator's documents: [{"id": ..., "text": ...}, ...]."""
    with open(path) as fh:
        return json.load(fh)


def load_questions(path):
    """Read the quiz. Each entry carries the doc ids that support the answer,
    which lets a miss be attributed to retrieval rather than to the model."""
    with open(path) as fh:
        return json.load(fh)


def rank_chunks(question, docs, k=3, expand=True):
    """Return up to k docs relevant to the question, by TF-IDF cosine similarity.

    A single pass only reaches docs sharing vocabulary with the question, which
    multi-hop questions don't: "who worked at the place Nina visited" has no
    overlap with the doc naming Helen. With expand=True a second pass re-scores
    against the question plus whatever the first pass found - once "the pharmacy"
    is in the query, the doc naming Helen becomes reachable.

    expand=False is the weaker single-pass baseline, kept so the two can be
    compared directly.

    Only docs with nonzero similarity are returned, so a thin result stays thin
    rather than being padded with arbitrary zero-scoring docs.

    One chunk per doc - the generator emits a sentence each. Swapping TF-IDF for
    embeddings later means rewriting this function and nothing else.
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


def all_chunks(question, docs, k=None, expand=None):
    """Full-context arm: no retrieval, every doc goes to the model.

    Same signature as rank_chunks so the two are interchangeable.

    This is the control, but it is NOT an upper bound - on the five-doc toy set it
    already lost to the text arm on a two-hop question, and its answer moved with
    the order of the documents. Whatever it scores, some of it is doc ordering
    rather than the model's ability to use the context.
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


ARTICLES = ("the ", "a ", "an ")


def normalize(text):
    """Fold the differences that shouldn't count as a wrong answer.

    Leading articles go too: the generator writes "the station" and "a book", and
    a model answering "station" is right about retrieval, which is what's measured.
    """
    text = text.strip().lower().rstrip(".").strip()
    for article in ARTICLES:
        if text.startswith(article):
            return text[len(article):]
    return text


def format_ids(chunks, limit=6):
    """Doc ids, abbreviated - the full-context arm gets long as the corpus grows."""
    ids = [c["id"] for c in chunks]
    if len(ids) > limit:
        return f"{', '.join(ids[:limit])} +{len(ids) - limit} more"
    return ", ".join(ids) if ids else "(nothing retrieved)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", default="data/docs.json")
    parser.add_argument("--questions", default="data/questions.json")
    # k=8 on a 50-doc corpus. Chosen empirically: k=3 and k=5 both fail to retrieve
    # the second doc of a two-hop question. This will need revisiting as the corpus
    # grows, and the graph arm needs an equivalent depth knob or the arms aren't
    # comparable.
    parser.add_argument("--k", type=int, default=8, help="chunks the text arm retrieves")
    parser.add_argument(
        "--no-expand",
        dest="expand",
        action="store_false",
        help="disable the text arm's second retrieval pass (query expansion)",
    )
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
            # Separates "retrieval missed it" from "model had it and blew it".
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

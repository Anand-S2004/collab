import json

import ollama

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


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


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

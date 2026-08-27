import argparse
import statistics
import time

import ollama

from common import MODEL, PROMPT, load_json
from graph_arm import graph_retrieve
from text_arm import all_chunks, rank_chunks


def timed_ask(question, chunks):
    context = "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    prompt = PROMPT.format(context=context, question=question)
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "seed": 0},
    )
    ns = 1_000_000.0
    return {
        "prompt_tokens": resp.get("prompt_eval_count", 0),
        "gen_tokens": resp.get("eval_count", 0),
        "prompt_ms": resp.get("prompt_eval_duration", 0) / ns,
        "gen_ms": resp.get("eval_duration", 0) / ns,
        "total_ms": resp.get("total_duration", 0) / ns,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    docs = load_json("data/docs.json")
    kg = load_json("data/kg.json")
    questions = load_json("data/questions.json")

    arms = [
        ("text", lambda q: rank_chunks(q, docs, k=args.k, expand=True)),
        ("text-noexpand", lambda q: rank_chunks(q, docs, k=args.k, expand=False)),
        ("graph", lambda q: graph_retrieve(q, kg, k=args.k, resolve=True)),
        ("full-context", lambda q: all_chunks(q, docs)),
    ]

    print("warming up model")
    timed_ask("warmup", [{"id": "d0", "text": "warmup."}])

    stats = {}
    for name, retrieve in arms:
        retr, prompt_ms, gen_ms, total_ms, ptok, chunks_n = [], [], [], [], [], []
        for item in questions:
            for _ in range(args.reps):
                t0 = time.perf_counter()
                chunks = retrieve(item["q"])
                retr.append((time.perf_counter() - t0) * 1000)
                m = timed_ask(item["q"], chunks)
                prompt_ms.append(m["prompt_ms"])
                gen_ms.append(m["gen_ms"])
                total_ms.append(m["total_ms"])
                ptok.append(m["prompt_tokens"])
                chunks_n.append(len(chunks))
        stats[name] = {
            "retrieval_ms": statistics.mean(retr),
            "prompt_ms": statistics.mean(prompt_ms),
            "gen_ms": statistics.mean(gen_ms),
            "model_ms": statistics.mean(total_ms),
            "prompt_tokens": statistics.mean(ptok),
            "chunks": statistics.mean(chunks_n),
        }

    print(f"\n{len(docs)} docs, {len(questions)} questions, {args.reps} reps, k={args.k}\n")
    head = f"{'arm':<14}{'chunks':>7}{'ptokens':>9}{'retrieval':>11}{'model':>9}{'total':>9}{'x fastest':>11}"
    print(head)
    print("-" * len(head))
    fastest = min(s["retrieval_ms"] + s["model_ms"] for s in stats.values())
    for name, s in stats.items():
        total = s["retrieval_ms"] + s["model_ms"]
        print(f"{name:<14}{s['chunks']:>7.0f}{s['prompt_tokens']:>9.0f}"
              f"{s['retrieval_ms']:>10.1f}m{s['model_ms']:>8.0f}m{total:>8.0f}m{total/fastest:>10.2f}x")
    print("\nretrieval breakdown (ms), and model split")
    for name, s in stats.items():
        share = 100 * s["retrieval_ms"] / (s["retrieval_ms"] + s["model_ms"])
        print(f"  {name:<14} retrieval {s['retrieval_ms']:>7.2f} ms ({share:.2f}% of total)"
              f"   prompt-eval {s['prompt_ms']:>6.0f} ms   generate {s['gen_ms']:>6.0f} ms")


if __name__ == "__main__":
    main()

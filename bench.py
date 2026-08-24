import argparse

from common import ask_model, format_ids, load_json, normalize
from graph_arm import graph_retrieve
from text_arm import all_chunks, rank_chunks


def build_arms(docs, kg, k, expand, resolve):
    return [
        ("text", lambda q: rank_chunks(q, docs, k=k, expand=expand)),
        ("text-noexpand", lambda q: rank_chunks(q, docs, k=k, expand=False)),
        ("graph", lambda q: graph_retrieve(q, kg, k=k, resolve=resolve)),
        ("full-context", lambda q: all_chunks(q, docs)),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/docs.json")
    parser.add_argument("--kg", default="data/kg.json")
    parser.add_argument("--questions", default="data/questions.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--no-expand", dest="expand", action="store_false")
    parser.add_argument("--no-resolve", dest="resolve", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    docs = load_json(args.docs)
    kg = load_json(args.kg)
    questions = load_json(args.questions)
    arms = build_arms(docs, kg, args.k, args.expand, args.resolve)

    print(f"{len(docs)} docs, {len(kg['claims'])} claims, {len(questions)} questions, k={args.k}")
    print(f"entity resolution: {'on' if args.resolve else 'off'}\n")

    correct = {name: 0 for name, _ in arms}
    support = {name: 0 for name, _ in arms}

    for item in questions:
        if not args.quiet:
            print(f"{item['id']} ({item['hops']}-hop): {item['q']}")
            print(f"   expected: {item['answer']}   supported by: {', '.join(item['docs'])}")
        for name, retrieve in arms:
            chunks = retrieve(item["q"])
            answer = ask_model(item["q"], chunks)

            hit = normalize(answer) == normalize(item["answer"])
            missing = set(item["docs"]) - {c["id"] for c in chunks}
            correct[name] += hit
            support[name] += not missing

            if not args.quiet:
                flag = "got support" if not missing else f"MISSING {','.join(sorted(missing))}"
                print(f"   {name:<14} {'PASS' if hit else 'FAIL'}  {answer!r}")
                print(f"   {'':<14} {flag} [{format_ids(chunks)}]")
        if not args.quiet:
            print()

    n = len(questions)
    print("=" * 58)
    print(f"{'arm':<14} {'answers':<10} {'retrieved support':<18}")
    print("-" * 58)
    for name, _ in arms:
        print(f"{name:<14} {correct[name]}/{n:<8} {support[name]}/{n}")


if __name__ == "__main__":
    main()

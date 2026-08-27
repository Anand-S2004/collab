import re

from common import load_json


def load_kg(path):
    return load_json(path)


def resolve_entities(kg):
    canonical = {}
    alias = {}
    for e in kg["entities"]:
        key = (e["type"], e["name"].lower())
        if key not in canonical:
            canonical[key] = e["id"]
        alias[e["id"]] = canonical[key]
    return alias


def identity_alias(kg):
    return {e["id"]: e["id"] for e in kg["entities"]}


def verbalize(claim, names):
    subject = names[claim["subject"]]
    predicate = claim["predicate"].lower().replace("_", " ")
    obj = names[claim["object"]]
    return f"{subject} {predicate} {obj}."


def graph_retrieve(question, kg, k=8, hops=2, resolve=True):
    alias = resolve_entities(kg) if resolve else identity_alias(kg)

    names = {}
    for e in kg["entities"]:
        names[alias[e["id"]]] = e["name"]

    claims = []
    for c in kg["claims"]:
        claims.append({
            "id": c["id"],
            "doc": c["provenance"]["document_id"],
            "subject": alias[c["subject"]],
            "object": alias[c["object"]],
            "predicate": c["predicate"],
        })

    lowered = question.lower()
    frontier = set()
    for node, name in names.items():
        if re.search(rf"\b{re.escape(name.lower())}\b", lowered):
            frontier.add(node)

    def relevance(claim):
        words = claim["predicate"].lower().replace("_", " ").split()
        return sum(1 for w in words if re.search(rf"\b{re.escape(w)}\b", lowered))

    beam = max(1, k // hops)
    found = []
    seen = set()
    for _ in range(hops):
        step = []
        for c in claims:
            if c["id"] in seen:
                continue
            if c["subject"] in frontier or c["object"] in frontier:
                step.append(c)
        if not step:
            break
        step.sort(key=relevance, reverse=True)
        step = step[:beam]
        for c in step:
            seen.add(c["id"])
            found.append(c)
            frontier.add(c["subject"])
            frontier.add(c["object"])

    return [{"id": c["doc"], "text": verbalize(c, names)} for c in found[:k]]


def validate_kg(kg):
    problems = []
    ids = set()
    by_name = {}

    for e in kg.get("entities", []):
        if not all(key in e for key in ("id", "type", "name")):
            problems.append(f"entity missing id/type/name: {e}")
            continue
        if not isinstance(e["name"], str):
            problems.append(f"entity {e['id']} name is not a string: {e['name']!r}")
            continue
        if e["id"] in ids:
            problems.append(f"duplicate entity id: {e['id']}")
        ids.add(e["id"])

        raw = e["name"]
        if raw != raw.strip() or "  " in raw:
            problems.append(f"entity {e['id']} name has stray whitespace: {raw!r}")
        if not raw.isascii():
            problems.append(f"entity {e['id']} name is non-ascii, may not match variants: {raw!r}")

        by_name.setdefault(raw.lower(), set()).add(e["type"])

    for name, types in by_name.items():
        if len(types) > 1:
            problems.append(f"name {name!r} carries multiple types {sorted(types)}, will not merge")

    stripped = {}
    for name in by_name:
        bare = name
        for art in ("the ", "a ", "an "):
            if bare.startswith(art):
                bare = bare[len(art):]
                break
        stripped.setdefault(bare, []).append(name)
    for bare, variants in stripped.items():
        if len(variants) > 1:
            problems.append(f"names differ only by article, will not merge: {sorted(variants)}")

    for c in kg.get("claims", []):
        for slot in ("subject", "object"):
            if c.get(slot) not in ids:
                problems.append(f"claim {c.get('id')} {slot} points at unknown entity {c.get(slot)!r}")

    return problems

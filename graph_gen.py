"""This code will generate a random KG and its equivalent text.
The KG is the source of truth.
Use different seed values for different paragraphs. same templates
"""
import argparse
import json
import random
from pathlib import Path
OUT = Path("data")
NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Helen",
    "Ivan",
    "Julia",
    "Kevin",
    "Laura",
    "Michael",
    "Nina",
    "Oscar",
    "Paul",
    "Rachel",
    "Sam",
    "Tina",
    "Victor",
    "Wendy",
]

OBJECTS = [
    "milk",
    "bread",
    "coffee",
    "the dog",
    "the cat",
    "the newspaper",
    "a book",
    "a package",
    "groceries",
    "a bicycle",
    "the car",
    "lunch",
]


PLACES = [
    "the store",
    "the park",
    "the library",
    "the office",
    "the cafe",
    "the station",
    "the pharmacy",
    "the market",
]

ACTION_TEMPLATES = [
    {
        "id": "buy",
        "predicate": "BOUGHT",
        "object_type": "object",
        "sentence": "{subject} went to buy {object}.",
    },
    {
        "id": "walk",
        "predicate": "WALKED",
        "object_type": "object",
        "sentence": "{subject} decided to walk {object}.",
    },
    {
        "id": "read",
        "predicate": "READ",
        "object_type": "object",
        "sentence": "{subject} spent the afternoon reading {object}.",
    },
    {
        "id": "visit",
        "predicate": "VISITED",
        "object_type": "place",
        "sentence": "{subject} visited {object}.",
    },
    {
        "id": "drive",
        "predicate": "DROVE_TO",
        "object_type": "place",
        "sentence": "{subject} drove to {object}.",
    },
    {
        "id": "work",
        "predicate": "WORKED_AT",
        "object_type": "place",
        "sentence": "{subject} worked at {object}.",
    },
]

def make_entity(entity_id, entity_type, name):
    return {
        "id": entity_id,
        "type": entity_type,
        "name": name,
    }
def generate(seed, num_sentences):
    """Generate one random KG and its equivalent text."""
    rng = random.Random(seed)
    entities = []
    claims = []
    documents = []
    #randomly sample a subset
    num_people = min(
        len(NAMES),
        max(2, num_sentences // 5),
    )
    selected_names = rng.sample(NAMES, num_people)
    name_to_id = {}
    for i, name in enumerate(selected_names):
        entity_id = f"e{i + 1}"
        entities.append(
            make_entity(
                entity_id,
                "person",
                name,
            )
        )
        name_to_id[name] = entity_id
#generate sentences
    for i in range(num_sentences):
        template = rng.choice(ACTION_TEMPLATES)
        subject_name = rng.choice(selected_names)
        if template["object_type"] == "object":
            object_name = rng.choice(OBJECTS)
            object_type = "object"
        elif template["object_type"] == "place":
            object_name = rng.choice(PLACES)
            object_type = "place"
        else:
            raise ValueError(
                f"Unknown object type: {template['object_type']}"
            )
        # Create an entity for the object/place.
        object_id = f"e_obj_{i + 1}"
        entities.append(
            make_entity(
                object_id,
                object_type,
                object_name,
            )
        )

        subject_id = name_to_id[subject_name]
        claim_id = f"c{i + 1}"
        document_id = f"d{i + 1}"
        claim = {
            "id": claim_id,
            "claim_type": "event",
            "subject": subject_id,
            "predicate": template["predicate"],
            "object": object_id,
            "temporal_scope": {
                "valid_from": None,#temporal scope is left blank on purpose for now
                "valid_to": None,
            },
            "semantic_scope": {},
            "provenance": {
                "document_id": document_id,
            },
            "confidence": 1.0,
        }

        sentence = template["sentence"].format(
            subject=subject_name,
            object=object_name,
        )

        document = {
            "id": document_id,
            "text": sentence,
        }

        claims.append(claim)
        documents.append(document)

    return {
        "entities": entities,
        "claims": claims,
    }, documents

def write_json(filename, data):
    path = OUT / filename
    path.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {path}")
def write_paragraph(documents):
    """Combine all generated sentences into one paragraph."""
    return " ".join(doc["text"] for doc in documents)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    parser.add_argument(
        "--sentences",
        type=int,
        default=50,
        help="Number of sentences to generate.",
    )
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    kg, documents = generate(
        seed=args.seed,
        num_sentences=args.sentences,
    )
    paragraph = write_paragraph(documents)
    write_json("kg.json", kg)
    write_json("docs.json", documents)
    (OUT / "para.txt").write_text(
        paragraph + "\n",
        encoding="utf-8",
    )
if __name__ == "__main__":
    main()
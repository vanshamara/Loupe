"""Embed a small arXiv corpus into JSONL for the Loupe benchmark.

Fetches recent papers in several categories from the public arXiv API, embeds
their title and abstract with a local sentence-transformers model, and writes
base and query JSONL files into datasets/arxiv/. No API key is needed.

Usage:
    pip install -r scripts/requirements.txt
    python scripts/embed_arxiv.py
"""

import json
import os
import random
import time
import urllib.request
import xml.etree.ElementTree as ElementTree

from sentence_transformers import SentenceTransformer

CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "cs.DB", "stat.ML", "math.PR"]
PER_CATEGORY = 300
QUERY_COUNT = 100
MODEL_NAME = "all-MiniLM-L6-v2"
OUT_DIR = "datasets/arxiv"

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


def fetch_category(category, count):
    """Returns the raw Atom response for the newest `count` papers in a category."""
    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=cat:{category}&start=0&max_results={count}"
        "&sortBy=submittedDate&sortOrder=descending"
    )
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def parse_entries(xml_bytes):
    """Pulls (text, primary_category, year, doc_type) out of an Atom response."""
    root = ElementTree.fromstring(xml_bytes)
    entries = []
    for entry in root.findall(f"{ATOM}entry"):
        title = (entry.findtext(f"{ATOM}title") or "").strip()
        summary = (entry.findtext(f"{ATOM}summary") or "").strip()
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if len(published) >= 4 else 0
        primary = entry.find(f"{ARXIV}primary_category")
        category = primary.get("term") if primary is not None else "unknown"
        doc_type = "primary" if len(entry.findall(f"{ATOM}category")) <= 1 else "cross"
        text = f"{title}. {summary}".replace("\n", " ")
        entries.append((text, category, year, doc_type))
    return entries


def write_jsonl(path, meta, embeddings):
    """Writes one record per line with its embedding and metadata."""
    with open(path, "w") as out:
        for vector, (_, category, year, doc_type) in zip(embeddings, meta):
            record = {
                "embedding": [round(float(value), 6) for value in vector],
                "category": category,
                "year": year,
                "doc_type": doc_type,
            }
            out.write(json.dumps(record) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    collected = []
    for category in CATEGORIES:
        print(f"fetching {category} ...")
        collected.extend(parse_entries(fetch_category(category, PER_CATEGORY)))
        time.sleep(3)

    print(f"embedding {len(collected)} papers with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [text for text, _, _, _ in collected]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    random.seed(0)
    order = list(range(len(collected)))
    random.shuffle(order)
    collected = [collected[i] for i in order]
    embeddings = embeddings[order]

    queries, query_vectors = collected[:QUERY_COUNT], embeddings[:QUERY_COUNT]
    base, base_vectors = collected[QUERY_COUNT:], embeddings[QUERY_COUNT:]

    write_jsonl(f"{OUT_DIR}/arxiv_base.jsonl", base, base_vectors)
    write_jsonl(f"{OUT_DIR}/arxiv_query.jsonl", queries, query_vectors)
    print(f"wrote {len(base)} base and {len(queries)} query records to {OUT_DIR}/")


if __name__ == "__main__":
    main()

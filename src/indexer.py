"""Embed and persist chunks in separate Chroma collections."""
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import chromadb
from .config import CHROMA_DIR, CORPUS_DIR, EMBEDDING_MODEL, ensure_dirs
from .pdf_ingest import extract_pages, make_chunks
from dataclasses import asdict

def embed_texts(client, texts, model, batch_size=64):
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend([x.embedding for x in response.data])
    return vectors

def index_strategy(pdf_path: str, strategy: str):
    ensure_dirs()
    client = OpenAI()
    pages = extract_pages(pdf_path)
    chunks = make_chunks(pages, strategy)
    if not chunks:
        raise RuntimeError("Chunking produced no content.")
    vectors = embed_texts(client, [c.text for c in chunks], EMBEDDING_MODEL)
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    name = f"ncci_2026_{strategy}"
    try:
        db.delete_collection(name)
    except Exception:
        pass
    collection = db.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 400):
        batch = chunks[i:i+400]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=vectors[i:i+400],
            metadatas=[{"page_start": c.page_start, "page_end": c.page_end,
                        "section": c.section, "strategy": c.strategy} for c in batch]
        )
    corpus_file = CORPUS_DIR / f"{strategy}.jsonl"
    with corpus_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            with corpus_file.open("w", encoding="utf-8") as f:
                for c in chunks:
                    json_line = json.dumps(
                        asdict(c),
                        ensure_ascii=False
                    )
                    f.write(json_line + "\n")            
    print(f"Indexed {len(chunks)} chunks into {name}; corpus: {corpus_file}")

def main():
    p = argparse.ArgumentParser(description="Ingest the CMS NCCI PDF.")
    p.add_argument("--pdf", required=True, help="Path to locally downloaded 2026 NCCI PDF")
    p.add_argument("--strategy", choices=["fixed", "structure", "all"], default="all")
    args = p.parse_args()
    strategies = ["fixed", "structure"] if args.strategy == "all" else [args.strategy]
    for strategy in strategies:
        index_strategy(args.pdf, strategy)

if __name__ == "__main__":
    main()

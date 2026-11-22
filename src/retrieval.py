"""Vector and hybrid retrieval with reciprocal-rank fusion."""
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import chromadb
from openai import OpenAI
from .config import CHROMA_DIR, CORPUS_DIR, EMBEDDING_MODEL

def tokenize(text):
    import re
    return re.findall(r"[a-z0-9]+", text.lower())

class Retriever:
    def __init__(self, strategy="structure"):
        self.strategy = strategy
        self.db = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.db.get_collection(f"ncci_2026_{strategy}")
        self.client = OpenAI()
        self._load_bm25()


    def _load_bm25(self):
        path = CORPUS_DIR / f"{self.strategy}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {path}\n"
                "Please run the PDF ingestion process first."
            )

        self.corpus = []

        with path.open("r", encoding="utf-8") as f:
            content = f.read()

        decoder = json.JSONDecoder()
        position = 0

        while position < len(content):
            # Ignore whitespace and newline characters.
            while position < len(content) and content[position].isspace():
                position += 1

            if position >= len(content):
                break

            try:
                obj, end_position = decoder.raw_decode(
                    content,
                    position
                )

                if not isinstance(obj, dict):
                    raise ValueError(
                        f"Expected JSON object at character {position}"
                    )

                self.corpus.append(obj)
                position = end_position

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, near character "
                    f"{position}: {exc}"
                ) from exc

        if not self.corpus:
            raise ValueError(
                f"No valid chunks found in {path}"
            )

        required_fields = {
            "id", "text", "page_start", "page_end"
        }

        for index, item in enumerate(self.corpus):
            missing = required_fields - item.keys()

            if missing:
                raise ValueError(
                    f"Chunk {index} is missing fields: {missing}"
                )

        self.bm25 = BM25Okapi([
            tokenize(item["text"])
            for item in self.corpus
        ])

        print(
            f"Loaded {len(self.corpus)} chunks "
            f"from {path.name}"
        )


    def vector_search(self, question, k=6):
        qvec = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[question]).data[0].embedding
        result = self.collection.query(query_embeddings=[qvec], n_results=k,
                                       include=["documents", "metadatas", "distances"])
        hits = []
        for i, doc in enumerate(result["documents"][0]):
            meta = result["metadatas"][0][i]
            hits.append({"text": doc, **meta, "distance": result["distances"][0][i],
                         "id": result["ids"][0][i], "retrieval": "vector"})
        return hits

    def bm25_search(self, question, k=6):
        scores = self.bm25.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [{**self.corpus[i], "score": float(scores[i]), "retrieval": "bm25"} for i in order]

    def search(self, question, mode="vector", k=6):
        if mode == "vector":
            return self.vector_search(question, k)
        if mode == "bm25":
            return self.bm25_search(question, k)
        if mode == "hybrid":
            # Reciprocal Rank Fusion; avoids mixing incomparable raw scores.
            v = self.vector_search(question, k * 2)
            b = self.bm25_search(question, k * 2)
            fused = {}
            for rank, hit in enumerate(v, 1):
                fused.setdefault(hit["id"], {"hit": hit, "rrf": 0.0})
                fused[hit["id"]]["rrf"] += 1 / (60 + rank)
            for rank, hit in enumerate(b, 1):
                fused.setdefault(hit["id"], {"hit": hit, "rrf": 0.0})
                fused[hit["id"]]["rrf"] += 1 / (60 + rank)
            ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)[:k]
            return [{**x["hit"], "rrf": x["rrf"], "retrieval": "hybrid"} for x in ranked]
        raise ValueError("mode must be vector, bm25, or hybrid")

"""
Retriever — searches ChromaDB for chunks relevant to a question.
Usage:  python src/retriever.py
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths & config ────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")
EMBED_MODEL = "all-MiniLM-L6-v2"  # must match ingest.py exactly
COLLECTION = "documents"  # must match ingest.py exactly


def retrieve(question: str, n_results: int = 3) -> list[dict]:
    """
    Embed a question and return the n most relevant chunks from ChromaDB.
    Each returned dict has: text, source, page, distance.
    """

    # Step 1 — embed the question using the same model used during ingestion
    model = SentenceTransformer(EMBED_MODEL)
    question_embedding = model.encode(question).tolist()

    # Step 2 — connect to ChromaDB and open the collection
    db = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    collection = db.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Step 3 — search for the most similar chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Step 4 — reformat ChromaDB's nested response into a clean list
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
            "type": results["metadatas"][0][i]["type"],
            # Distance in cosine similarity: closer to 0.0 means more similar, closer to 2.0 means unrelated.
        })

    return chunks

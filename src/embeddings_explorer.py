from sentence_transformers import SentenceTransformer
import chromadb
import os

# ── Load embedding model ──────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
# First run downloads ~80MB — subsequent runs load from cache instantly
print("Model loaded.\n")


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: What does an embedding actually look like?
# ═══════════════════════════════════════════════════════════════════════════
def experiment_1_what_is_an_embedding():
    print("=" * 55)
    print("EXPERIMENT 1: What is an embedding?")
    print("=" * 55)

    sentence = "Machine learning is a subset of artificial intelligence."
    vector = embedder.encode(sentence)

    print(f"Sentence : {sentence}")
    print(f"Vector   : {vector[:5]} ...")  # show first 5 numbers
    print(f"Length   : {len(vector)} numbers")
    print("""
Insight: Every sentence becomes a fixed-length list of numbers.
Similar sentences produce similar number patterns.
This is how meaning gets converted to math.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Semantic similarity — meaning beats keywords
# ═══════════════════════════════════════════════════════════════════════════
def experiment_2_semantic_similarity():
    print("=" * 55)
    print("EXPERIMENT 2: Semantic similarity")
    print("=" * 55)

    from sklearn.metrics.pairwise import cosine_similarity

    base = "What is machine learning?"

    comparisons = [
        "Explain ML to me",  # same meaning, different words
        "What is deep learning?",  # related topic
        "How do I bake a chocolate cake?",  # completely unrelated
        "Describe artificial intelligence concepts",  # loosely related
    ]

    base_vector = embedder.encode([base])

    print(f"Base question: '{base}'\n")
    print(f"{'Comparison sentence':<45} {'Similarity':>10}")
    print("─" * 57)

    for sentence in comparisons:
        vec = embedder.encode([sentence])
        score = cosine_similarity(base_vector, vec)[0][0]
        bar = "█" * int(score * 20)
        print(f"{sentence:<45} {score:>6.3f}  {bar}")

    print("""
Insight: 'Explain ML to me' scores high even though it shares
no words with 'What is machine learning?' — the embedding
model understands they mean the same thing.
This is why RAG finds the right document chunks even when
the user's question uses different words than the document.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Store and search in ChromaDB
# ═══════════════════════════════════════════════════════════════════════════
def experiment_3_chromadb_search():
    print("=" * 55)
    print("EXPERIMENT 3: ChromaDB storage and search")
    print("=" * 55)

    # Create ChromaDB client — persists data to disk in vectorstore/
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VECTORESTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")
    os.makedirs(VECTORESTORE_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=VECTORESTORE_PATH)

    collection = db.get_or_create_collection("experiment")

    # Sample sentences — imagine these are chunks from a PDF
    documents = [
        "Python is a high-level programming language known for simplicity.",
        "Machine learning models learn patterns from training data.",
        "ChromaDB is a vector database for storing embeddings.",
        "Neural networks are inspired by the human brain structure.",
        "FastAPI is a modern Python framework for building REST APIs.",
        "Embeddings convert text into numerical vectors for semantic search.",
        "Supervised learning uses labeled data to train predictive models.",
        "Docker containers package applications with their dependencies.",
    ]

    # Embed all documents and store in ChromaDB
    print("Storing documents in ChromaDB...")
    embeddings = embedder.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    print(f"Stored {len(documents)} documents.\n")

    # Now search with questions
    queries = [
        "How do AI models get trained?",
        "What tools help deploy applications?",
        "Tell me about text to number conversion",
        "How does embedding get stored in ChromaDB?"
    ]

    for query in queries:
        print(f"Query: '{query}'")
        query_embedding = embedder.encode([query]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=2  # return top 2 most similar documents
        )

        print("Top matches:")
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # convert distance to similarity score
            print(f"  {i + 1}. [{similarity:.3f}] {doc}")
        print()

    print("""
Insight: ChromaDB found relevant documents even though
the query words didn't match the stored text exactly.
'How do AI models get trained?' found the ML sentence —
no shared keywords, just shared meaning.
This is the core of what makes RAG work.
    """)


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Keyword search vs Semantic search — side by side
# ═══════════════════════════════════════════════════════════════════════════
def experiment_4_keyword_vs_semantic():
    print("=" * 55)
    print("EXPERIMENT 4: Keyword search vs Semantic search")
    print("=" * 55)

    documents = [
        "The automobile industry is shifting towards electric vehicles.",
        "Python programming is popular for data science projects.",
        "Cars and trucks are common forms of road transportation.",
        "Machine learning is transforming how software is built.",
        "Vehicles powered by batteries are becoming more affordable.",
    ]

    query = "What is happening with electric cars?"

    # ── Keyword search (old way) ──
    print(f"Query: '{query}'\n")
    print("KEYWORD SEARCH results (simple word matching):")
    keyword_hits = [
        doc for doc in documents
        if any(word in doc.lower() for word in query.lower().split())
    ]
    if keyword_hits:
        for doc in keyword_hits:
            print(f"  ✓ {doc}")
    else:
        print("  No results found.")

    # ── Semantic search (our way) ──
    print("\nSEMANTIC SEARCH results (meaning matching):")
    query_vec = embedder.encode([query]).tolist()
    doc_vecs = embedder.encode(documents).tolist()

    from sklearn.metrics.pairwise import cosine_similarity
    scores = cosine_similarity([query_vec[0]], doc_vecs)[0]
    ranked = sorted(zip(scores, documents), reverse=True)

    for score, doc in ranked[:3]:
        print(f"  [{score:.3f}] {doc}")

    print("""
Insight: Keyword search missed 'Vehicles powered by batteries'
because it doesn't contain the word 'cars' or 'electric'.
Semantic search found it because the MEANING is the same.
This is why your document chatbot will feel intelligent —
not just a fancy Ctrl+F.
    """)


# ── Run all experiments ───────────────────────────────────────────────────
if __name__ == "__main__":
    experiment_1_what_is_an_embedding()
    experiment_2_semantic_similarity()
    experiment_3_chromadb_search()
    experiment_4_keyword_vs_semantic()

    print("=" * 55)
    print("Day 3 Complete.")
    print("You now understand:")
    print("  ✅ What an embedding is")
    print("  ✅ How semantic similarity works")
    print("  ✅ How to store and search in ChromaDB")
    print("  ✅ Why semantic search beats keyword search")
    print("\nNext: Day 4 — Load a real PDF and build the ingestion pipeline")
    print("=" * 55)

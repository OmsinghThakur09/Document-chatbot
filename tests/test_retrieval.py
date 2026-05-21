from src import retriever


def test_retrieve():
    test_questions = [
        "What is this document about?",
        "Who is the author?",
        "What are the main topics covered?",
    ]

    print("=" * 50)
    print("Retrieval Test")
    print("=" * 50)

    for question in test_questions:
        chunks = retriever.retrieve(question)
        print(f"\n❓ Question: {question}")
        chunks = retriever.retrieve(question)
        for i, chunk in enumerate(chunks, 1):
            print('-'*50)
            print(f"  Result {i} | Page {chunk['page']} | "
                  f"  Distance: {chunk['distance']:.3f}")
            print(f"  Preview: {chunk['text']}")
            print(f"  Length: {len(chunk["text"])}")


if __name__ == "__main__":
    test_retrieve()

# src/chatbot.py

import os
import ollama
from dotenv import load_dotenv

load_dotenv()

# ── LLM Backend Config ────────────────────────────────────────────────────────
MODEL = "gemma4:e4b"  # Ollama local model
GROQ_MODEL = "llama-3.1-8b-instant"  # Groq cloud model
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "groq"


def call_llm(messages: list[dict], temperature: float = 0.7) -> str:
    """
        Internal router — sends messages to either Ollama or Groq
        based on LLM_PROVIDER env variable.
        All other functions call this instead of ollama.chat() directly.
        Returns the assistant's reply as a plain string.
    """
    if LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    else:
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": temperature}
        )
        return response["message"]["content"]


# single question single answer function
def ask_gpt(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """
    basic function to ask a single question to LLM model
    :param prompt: user question
    :param system: this is for LLM model to tell its role, and it is already given as helpful assistant
    so no need to mention at the time of calling.
    :return: answer of user question in string format.

    ollama.chat() sends message to the local Ollama server
    at http://localhost:11434 — no internet, no API key, no limits.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    return call_llm(messages)


# multi-turn conversation
def chat_session():
    """
    this is similar function as above but here it answer multiple questions until user
    dont reply with 'quit'.
    to run a conversation it requires memory, model has no memory of its own, we give it
    memory by including the full history in each call manually.
    :return:
    """
    history = [
        {"role": "assistant", "content": "You are a concise helpful assistant."}
    ]
    print("chat started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break

        history.append({"role": "user", "content": user_input})  # appending user question to message call

        reply = call_llm(history)
        history.append({"role": "assistant", "content": reply})  # manually adding history to every call

        print(f"Assistant: {reply}\n")


def ask_with_usage(prompt: str) -> dict:
    """
    here we will get additional information with our answer that will help us with RAG
    and costing calculations.
    :param prompt: same user question
    :return: answer of question:
             input_tokens:
             output_tokens:
             total_tokens:
             cost_usd:
    """
    messages = [
        {"role": "system", "content": "You are a concise helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    answer = call_llm(messages)
    return {
        "answer": answer,
        # "input_tokens": answer.get("prompt_eval_count", 0),
        # "output_tokens": answer.get("eval_count", 0),
        # "total_tokens": answer.get("prompt_eval_count", 0) + answer.get("eval_count", 0),
        # "cost_usd": 0.0  # using local model so free
    }


def ask_with_context(question: str, chunks: list[dict], temperature: float = 0.7) -> str:
    """
    Answer a question using only the provided document chunks as context.
    chunks: list of dicts with keys — text, source, page, distance
    """

    # Edge case — no chunks retrieved
    if not chunks:
        return ("I couldn't find any relevant information in the "
                "document to answer that question.")

    # labelled context block from all retrieved chunks
    context_block = ""
    for i, chunk in enumerate(chunks, 1):
        context_block += (f"[Chunk {i} — Page {chunk['page']} "
                          f"from {chunk['source']}]\n")
        context_block += f"{chunk['text']}\n\n"

    # System prompt — instructs LLM to stay grounded in the document
    # Strict mode
    """system_prompt = You are a document assistant.
    Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I don't have enough information in the document to answer this."
    Do not make up information. Always mention which page your answer comes from."""

    # Hybrid mode
    system_prompt = """You are a knowledgeable document assistant with two sources of knowledge:
    1. The document context provided to you (always prioritise this)
    2. Your own general knowledge (use this to enrich and explain)

    Your answering approach:
    - Start by answering from the document context and cite the page number
    - Then expand the answer using your own knowledge to give more depth, examples, or explanation
    - Clearly separate the two parts using phrases like:
      "From the document (Page X):" and "Additionally from my knowledge:"
    - If the document has no relevant information at all, answer from your own knowledge
      and clearly say "This wasn't covered in the document, but from my knowledge:"
    - Never contradict what the document says — document always wins on conflicts
    - Give complete, helpful, detailed answers — not one-liners"""

    # User message — context + question together
    user_message = f"""Context from the document:
    {context_block}
    Question: {question}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return call_llm(messages, temperature)


def chat_session_with_context():
    """
    same as before chat_session function but now model has contex about document.
    :return: None
    """
    system_prompt = """You are a document assistant.
    Answer the user's question using ONLY the context provided below.
    If the answer is not in the context, say "I don't have enough information in the document to answer this."
    Do not make up information. Always mention which page your answer comes from."""

    history = [
        {"role": "assistant", "content": system_prompt}
    ]
    print("-" * 100)
    print("chat session has started. type 'quit' to exit\n")

    from retriever import retrieve

    while True:
        user_input = input('You:')
        if user_input.strip().lower() == 'quit':
            break

        chunks = retrieve(user_input)
        context_block = ""
        for i, chunk in enumerate(chunks):
            context_block += (f"[chunk {i} - page {chunk['page']}"
                              f"source {chunk['source']}]\n")
            context_block += f"{chunk['text']}\n\n"
        user_message = f"""Context from document:
        question: {user_input}
        {context_block}
        """
        history.append(
            {"role": "user", "content": user_message}
        )

        reply = call_llm(history)
        history.append(
            {"role": "assistant", "content": reply}
        )
        print(f"Assistant: {reply}\n")


# ── End-to-end RAG test ───────────────────────────────────────────────────────
def full_rag_test(question: str):
    """
    Runs the complete RAG pipeline for one question and
    compares RAG answer vs direct answer side by side.
    """
    from retriever import retrieve  # imported here to avoid circular imports

    print(f"\n{'=' * 55}")
    print(f"QUESTION: {question}")
    print(f"{'=' * 55}")

    # Retrieve relevant chunks
    chunks = retrieve(question)
    print(f"\n📚 Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  {i}. Page {chunk['page']} | "
              f"Distance: {chunk['distance']:.3f} | "
              f"Source: {chunk['source']}")

    # RAG answer — grounded in document
    print(f"\n🤖 RAG Answer (with document context):")
    rag_answer = ask_with_context(question, chunks)
    print(rag_answer)

    # Direct answer — no context, shows hallucination risk
    print(f"\n💬 Direct Answer (no document context):")
    direct_answer = ask_gpt(question)
    print(direct_answer)

    print(f"{'=' * 55}")


if __name__ == "__main__":
    # Change these questions to match your actual PDF content
    full_rag_test("What is this document about?")
    # full_rag_test("What are the main points covered?")
    # full_rag_test("Who is the author?")
    # full_rag_test("What are the benefits of Digital Detox?")

    # reply = ask_gpt("what is solar system?")
    # print(reply)

    # chat_session()
    # chat_session_with_context()

    # result = ask_with_usage("explain what is RAG in three sentence")
    # print(result)

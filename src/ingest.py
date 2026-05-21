"""
PDF Ingestion Pipeline
Usage:  python src/ingest.py --pdf data/yourfile.pdf
        python src/ingest.py --pdf data/yourfile.pdf --clear
"""

import os
import argparse
import time  # for storing additional info, time at which chunk get stored.
import chromadb
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ── Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore")
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# ── Config
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION = "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# step 0: Convert table into markdown
def table_to_markdown(table):
    """
    Convert a pdfplumber table (list of lists) to a markdown string.
    Empty cells are replaced with a dash so the structure is preserved.
    :param table: list of lists
    :return: markdown string
    """
    if not table:
        return ""

    rows = []
    for i, row in enumerate(table):

        cleaned = [str(cell).strip() if cell is not None else "-" for cell in row]  # replacing None with "-".
        rows.append("| " + " | ".join(cleaned) + " |")

        if i == 0:
            rows.append("|" + "|".join(["---"] * len(row)) + "|")

    return "\n".join(rows)


def merge_split_tables(blocks):
    """
    when the table continues on next page it needs to be merged with its previous half
    version so that the LLM will get the complete table block so that it wont misunderstand
    it as two separate and different tables.
    :param blocks:
    :return: merged table blocks
    """

    def column_count(md):
        first_line = md.strip().split("\n")[0]
        return first_line.count("|") - 1

    result = list(blocks)
    to_remove = set()

    for i in range(len(result)):
        if result[i]["type"] != "table" or i in to_remove:
            continue

        j = i + 1
        while j < len(result) and result[j]["type"] != "table":
            j += 1

        if j >= len(result) - 1:
            break

        a = result[i]
        b = result[j]

        #  Merge condition: consecutive pages and same no. of columns
        if b["page_number"] == a["page_number"] + 1 and column_count(a["text"]) == column_count(b["text"]):

            lines_b = b["text"].strip().split("\n")
            b_data_rows = [lines_b[0]] + lines_b[2:]

            merged_text = a["text"] + "\n" + "\n".join(b_data_rows)

            result[i] = {
                "text": merged_text,
                "page_number": a["page_number"],
                "type": "table",
                "source": a["source"],
            }

            to_remove.add(j)

    return [block for idx, block in enumerate(result) if idx not in to_remove]


# ── Step 1: Load PDF → list of (page_number, text)
def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF using pdfplumber.
    Tables are converted to markdown and kept as atomic blocks.
    Return a list of dicts: {text, page, type}
    """
    print(f"\n📄 Loading PDF: {pdf_path}")

    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(ROOT_DIR, pdf_path)
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    blocks = []  # each block = one chunk candidate
    filename = os.path.basename(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            # .bbox is the bounding box — a tuple of (x0, y0, x1, y1) coordinates in PDF units.
            table_bboxes = [tbl.bbox for tbl in page.find_tables()]  # Find bounding boxes of all tables on this page

            for table in page.extract_tables():
                md = table_to_markdown(table)
                if md.strip():
                    blocks.append(
                        {
                            "text": md,
                            "page_number": page_num,
                            "source": filename,
                            "type": "table",
                        })

            if table_bboxes:
                non_table_page = page
                for bbox in table_bboxes:
                    # pdfplumber: outside_bbox removes content inside that box
                    non_table_page = non_table_page.outside_bbox(bbox)
                    plain_text = non_table_page.extract_text()
            else:
                plain_text = page.extract_text()

            if plain_text and plain_text.strip():  # skip blank/image-only pages
                blocks.append({
                    "page_number": page_num,
                    "text": plain_text.strip(),
                    "source": filename,
                    "type": "text",
                })
            else:
                print(f"  ⚠️  Page {page_num} — no extractable text (image/blank), skipped")

    blocks = merge_split_tables(blocks)  # handling continued tables on consecutive pages

    print(f"  Extracted {len(blocks)} blocks "
          f"({sum(1 for b in blocks if b['type'] == 'table')} tables, "
          f"{sum(1 for b in blocks if b['type'] == 'text')} text)")
    return blocks


# ── Step 2: Chunk pages → smaller pieces
def chunk_pages(blocks: list[dict]) -> list[dict]:
    """Split blocks into overlapping chunks. Keeps source metadata."""
    print(f"\n  Chunking  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # tries to split on paragraphs → sentences → words → chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for block in blocks:
        if block["type"] == "table":
            chunks.append(block)
        else:
            sub_chunks = splitter.split_text(block["text"])
            for i, sc in enumerate(sub_chunks):
                chunks.append({
                    "text": sc,
                    "page_number": block["page_number"],
                    "source": block["source"],
                    "chunk_index": i,
                    "type": "text"
                })

    print(f"  ✅ {len(blocks)} pages  →  {len(chunks)} chunks"
          f"({sum(1 for c in chunks if c['type'] == 'table')} table chunks)")
    return chunks


#  Step 3: Embed + store in ChromaDB
def embed_and_store(chunks: list[dict], clear_first: bool = False) -> None:
    """Embed all chunks and upsert into ChromaDB."""
    print(f"\n Embedding with {EMBED_MODEL}  ({len(chunks)} chunks)")

    model = SentenceTransformer(EMBED_MODEL)

    # Extract just the text strings for batch embedding
    texts = [c["text"] for c in chunks]

    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    elapsed = time.time() - start
    print(f"  ✅ Embedded {len(embeddings)} chunks in {elapsed:.1f}s")

    #  ChromaDB setup
    db = chromadb.PersistentClient(path=VECTORSTORE_PATH)

    if clear_first:
        try:
            db.delete_collection(COLLECTION)
            print(f"  🗑️  Cleared existing collection '{COLLECTION}'")
        except Exception:
            pass  # collection didn't exist yet

    collection = db.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # Build IDs and metadata lists
    ids = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        # ID must be unique across all docs — combine source + page + chunk
        safe_source = chunk["source"].replace(".", "_").replace(" ", "_")
        ids.append(f"{safe_source}_p{chunk['page_number']}_c{i}")
        metadatas.append({
            "source": chunk["source"],
            "page_number": chunk["page_number"],
            "type": chunk['type']
        })

    # upsert = insert or update — safe to run multiple times
    collection.upsert(
        ids=ids,
        embeddings=[e.tolist() for e in embeddings],
        documents=texts,
        metadatas=metadatas,
    )

    count = collection.count()
    print(f"  ✅ ChromaDB now holds {count} total chunks in '{COLLECTION}'")


# Main
def ingest(pdf_path: str, clear_first: bool = False) -> None:
    print("=" * 55)
    print("  RAG INGESTION PIPELINE")
    print("=" * 55)

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)
    embed_and_store(chunks, clear_first=clear_first)

    print("\n🏁 Ingestion complete — PDF is now searchable!")
    print(f"   Vectorstore path: {VECTORSTORE_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Specify the path of PDF file")
    parser.add_argument("--clear", action="store_true", help="wipeout Collection before ingestion" )
    args = parser.parse_args()

    ingest(args.pdf, clear_first=args.clear)
    '''we used two user arguments "--pdf" and "--clear" and these arguments becomes "args.pdf" and "args.clear"
    that's why this line looks like this 'ingest(args.pdf, clear_first=args.clear)'.
    '''

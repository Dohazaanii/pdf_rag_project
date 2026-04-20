import os
import sys
import json
import argparse

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# ── CONFIG ───────────────────────────────────────────────
DEFAULT_DB_PATH  = "db"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"

# ── ARGS ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True,  help="User question")
parser.add_argument("--file",     required=False, help="Path to uploaded PDF or Word file (optional)")
args = parser.parse_args()

# ── EMBEDDINGS ───────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# ── LOAD VECTORSTORE ─────────────────────────────────────
def load_from_file(file_path: str) -> Chroma:
    """Build a temporary in-memory vector store from an uploaded file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".doc", ".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    # Use a temp DB path based on filename so it's reused across calls
    db_path = DEFAULT_DB_PATH + "_" + os.path.basename(file_path).replace(" ", "_")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    vectorstore.persist()
    return vectorstore


def load_default_db() -> Chroma:
    """Load the default pre-built vector DB."""
    if not os.path.exists(DEFAULT_DB_PATH) or not os.listdir(DEFAULT_DB_PATH):
        raise FileNotFoundError(
            "No default vector DB found. Please upload a document first."
        )
    return Chroma(persist_directory=DEFAULT_DB_PATH, embedding_function=embeddings)


# ── SELECT VECTORSTORE ───────────────────────────────────
try:
    if args.file and os.path.exists(args.file):
        vectorstore = load_from_file(args.file)
    else:
        vectorstore = load_default_db()
except FileNotFoundError as e:
    print(json.dumps({"answer": str(e), "pages": []}))
    sys.exit(0)

# ── RETRIEVER ────────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6}
)

# ── LLM ──────────────────────────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.1,
    num_predict=1024,
)

# ── PROMPT ───────────────────────────────────────────────
prompt_template = """
You are an expert document analyst. Your job is to give COMPLETE and DETAILED answers
based solely on the provided context.

STRICT RULES:
- Use ONLY the provided context — never hallucinate.
- Cover ALL relevant sections of the document, not just the first matching passage.
- If the answer spans multiple pages, synthesize ALL of them.
- If something is truly absent → say "Not mentioned in the document."
- Always cite page numbers when available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT (from {num_docs} document sections):
{context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

ANSWER FORMAT:
• Answer ONLY what was asked — nothing more, nothing less.
• Do NOT add summaries, overviews, or extra sections unless explicitly requested.
• Use bullet points only if the answer naturally has multiple items.
• Cite page numbers at the end.

ANSWER:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "num_docs"]
)

# ── RAG CHAIN ────────────────────────────────────────────
docs: list[Document] = retriever.invoke(args.question)

context = "\n\n---\n\n".join(
    f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
    for d in docs
)

filled_prompt = prompt.format(
    context=context,
    question=args.question,
    num_docs=len(docs)
)

response = llm.invoke(filled_prompt)
answer   = response.content if hasattr(response, "content") else str(response)
pages    = sorted(set(
    str(d.metadata.get("page", "?")) for d in docs
))

# ── OUTPUT (JSON for Laravel) ─────────────────────────────
print(json.dumps({
    "answer": answer,
    "pages":  pages,
}, ensure_ascii=False))
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── CONFIG ───────────────────────────────────────────────
PDF_PATH   = "docs/eng.pdf"
DB_PATH    = "db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── 1. LOAD / CREATE VECTOR STORE ────────────────────────
def load_or_create_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        print("✅ Loading existing vector DB...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    print("🔨 Creating new vector DB...")
    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()

    # 🔧 FIX 1 — larger chunks keep more context per passage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,       # was 700  → more content per chunk
        chunk_overlap=250,     # was 150  → smoother transitions
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    print(f"   📄 {len(pages)} pages → {len(chunks)} chunks")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    db.persist()
    return db


# ── 2. LLM ───────────────────────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0.1,   # slight creativity while staying factual
    num_predict=1024   # was 512 → allows longer, complete answers
)


# ── 3. PROMPT ────────────────────────────────────────────
# 🔧 FIX 2 — richer prompt that forces exhaustive answers
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


# ── 4. CUSTOM CHAIN WITH num_docs INJECTION ──────────────
# 🔧 FIX 3 — inject the actual number of retrieved docs into the prompt
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

class EnrichedRetrievalQA:
    """Thin wrapper that injects num_docs into the prompt."""

    def __init__(self, llm, retriever, prompt):
        self.llm       = llm
        self.retriever = retriever
        self.prompt    = prompt

    def invoke(self, query: str) -> dict:
        docs: list[Document] = self.retriever.invoke(query)

        context = "\n\n---\n\n".join(
            f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )

        filled_prompt = self.prompt.format(
            context=context,
            question=query,
            num_docs=len(docs)
        )

        response = self.llm.invoke(filled_prompt)
        answer   = response.content if hasattr(response, "content") else str(response)

        pages = sorted(set(d.metadata.get("page", "?") for d in docs))
        return {"result": answer, "source_documents": docs, "pages": pages}


# ── 5. SETUP ─────────────────────────────────────────────
vectorstore = load_or_create_db()

# 🔧 FIX 4 — k=8 fetches more candidates; fetch_k=20 for MMR diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6}
)

qa = EnrichedRetrievalQA(llm=llm, retriever=retriever, prompt=prompt)


# ── 6. CHAT LOOP ─────────────────────────────────────────
print("\n✅ Ready! Ask questions about your PDF. Type 'exit' to quit.\n")

try:
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("👋 Bye!")
            break
        if not question:
            continue

        result = qa.invoke(question)

        print("\n🤖 AI:")
        print(result["result"])
        print(f"\n📄 Sources (pages): {result['pages']}\n")
        print("─" * 60)

except KeyboardInterrupt:
    print("\n👋 Interrupted. Bye!")
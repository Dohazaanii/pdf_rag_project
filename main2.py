import os
import sys
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

# ── CONFIG ───────────────────────────────────────────────
PDF_PATH    = "docs/eng.pdf"
DB_PATH     = "db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── 1. LOAD / CREATE VECTOR STORE ────────────────────────
def load_or_create_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
        return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    loader = PyPDFLoader(PDF_PATH)
    pages  = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)

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
    temperature=0.1,
    num_predict=1024
)

# ── 3. PROMPT ────────────────────────────────────────────
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
- Answer ONLY what was asked — nothing more, nothing less.
- Use bullet points only if the answer naturally has multiple items.
- Cite page numbers at the end.

ANSWER:
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question", "num_docs"]
)

# ── 4. CUSTOM CHAIN ──────────────────────────────────────
class EnrichedRetrievalQA:
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
        pages    = sorted(set(d.metadata.get("page", "?") for d in docs))

        return {"result": answer, "source_documents": docs, "pages": pages}

# ── 5. SETUP ─────────────────────────────────────────────
vectorstore = load_or_create_db()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.6}
)

qa = EnrichedRetrievalQA(llm=llm, retriever=retriever, prompt=prompt)

# ── 6. MAIN ──────────────────────────────────────────────
if __name__ == "__main__":

    # Mode Laravel — reçoit la question en argument → retourne JSON
    if len(sys.argv) > 1:
        question = sys.argv[1]
        result   = qa.invoke(question)
        print(json.dumps({
            "answer": result["result"],
            "pages":  result["pages"]
        }, ensure_ascii=False))

    # Mode terminal — chat interactif
    else:
        print("\n✅ Ready! Type 'exit' to quit.\n")
        try:
            while True:
                question = input("You: ").strip()
                if question.lower() in ("exit", "quit"):
                    print("👋 Bye!")
                    break
                if not question:
                    continue
                result = qa.invoke(question)
                print("\n🤖 AI:", result["result"])
                print(f"📄 Pages: {result['pages']}\n")
                print("─" * 60)
        except KeyboardInterrupt:
            print("\n👋 Bye!")
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── App ────────────────────────────────────────────────────
app = FastAPI(title="PDF RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:63342"],   # In production, replace with your Vue dev server URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────
PDF_PATH = "docs/pfe_TA.pdf"

# ── Build RAG chain on startup ─────────────────────────────
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages.")

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks.")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating vector store...")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

llm = ChatOllama(model="llama3", temperature=0)

prompt_template = """
You are an expert assistant analyzing a technical project specification (Cahier des Charges)
for Port Tanger Alliance — a port operator on the Strait of Gibraltar.
The document describes a Final Year Engineering Project (PFE) to build an AI-powered
scheduling module using Large Language Models (LLMs) to automate port operator shift planning.

Use ONLY the information provided in the context below to answer the question.
If the answer is not found in the context, respond with:
"I don't have enough information in the document to answer that."
Do NOT invent or assume any information not present in the context.

━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT:
{context}
━━━━━━━━━━━━━━━━━━━━━━━━

QUESTION: {question}

INSTRUCTIONS:
- Answer in the same language as the question (French or English).
- For broad questions (e.g. "what is this about", "summarize", "context"), structure your answer as:
    📌 Project Summary
    🎯 Key Objectives
    🛠️ Tech Stack
    📅 Timeline & Deliverables
- For specific questions, answer directly and concisely.
- Cite the relevant section number (e.g. "Section 3.2") whenever possible.
- Use bullet points for lists, not long paragraphs.

ANSWER:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)

print("\n✅ RAG chain ready. API is starting...\n")

# ── Request / Response models ──────────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    source_pages: list[int | str]

# ── Endpoint ───────────────────────────────────────────────
@app.post("/ask", response_model=AnswerResponse)
def ask(body: QuestionRequest):
    result = qa_chain.invoke({"query": body.question})
    pages_used = sorted(set(
        d.metadata.get("page", "?") for d in result["source_documents"]
    ))
    return AnswerResponse(
        answer=result["result"],
        source_pages=pages_used
    )

@app.get("/health")
def health():
    return {"status": "ok"}
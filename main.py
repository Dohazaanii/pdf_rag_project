import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── 1. Config ──────────────────────────────────────────────
PDF_PATH = "docs/pfe_TA.pdf"

# ── 2. Load PDF ─────────────────────────────────────────────
print("Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages.")

# ── 3. Split text ───────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # ⬆ was 500 — larger chunks = more context per retrieval
    chunk_overlap=100     # ⬆ was 50  — more overlap = fewer cut-off sentences
)
chunks = splitter.split_documents(pages)
print(f"Split into {len(chunks)} chunks.")

# ── 4. Embeddings (LOCAL, FREE) ────────────────────────────
print("Loading embeddings...")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence tokenizer warning
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ── 5. Vector store ─────────────────────────────────────────
print("Creating vector store...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)

# ── 6. LLM (LOCAL via Ollama) ──────────────────────────────
llm = ChatOllama(
    model="llama3",
    temperature=0
)

# ── 7. Prompt Template ─────────────────────────────────────
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

# ── 8. RAG Chain ───────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # ⬆ was 3 — retrieve more chunks for broad questions
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # ✅ inject custom prompt
)

# ── 9. Chat loop ───────────────────────────────────────────
print("\n✅ Ready! Ask questions about your PDF. Type 'exit' to quit.\n")

while True:
    question = input("You: ").strip()
    if question.lower() == "exit":
        break
    if not question:
        continue

    result = qa_chain.invoke({"query": question})

    print(f"\nAI: {result['result']}")

    # ✅ Clean page citation (sorted list instead of raw set)
    pages_used = sorted(set(
        d.metadata.get("page", "?") for d in result["source_documents"]
    ))
    print(f"(Source pages: {', '.join(str(p) for p in pages_used)})\n")
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
    chunk_size=800,
    chunk_overlap=100
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
You are an expert assistant that answers questions based strictly on the content
of uploaded documents. You have no knowledge of the specific document in advance.

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
- Answer in the same language as the question (French, English, or any other language).
- For broad questions (e.g. "what is this about", "summarize", "context"), structure your answer as:
    📌 Document Summary
    🎯 Key Topics / Objectives
    🛠️ Methods / Tools / Technologies (if applicable)
    📅 Timeline / Structure (if applicable)
- For specific questions, answer directly and concisely.
- Cite the relevant section, chapter, or page number whenever possible.
- Use bullet points for lists, not long paragraphs.

ANSWER:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# ── 8. RAG Chain ───────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
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
    pages_used = sorted(set(
        d.metadata.get("page", "?") for d in result["source_documents"]
    ))
    print(f"(Source pages: {', '.join(str(p) for p in pages_used)})\n")










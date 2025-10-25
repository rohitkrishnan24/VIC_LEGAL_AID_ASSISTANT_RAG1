from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path
import os

# Configure paths
LEGAL_DIR = Path("data/legal_docs")
DB_DIR = Path("./chroma_langchain_db_legal")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# If the vector DB doesn't exist, (re)build it from legal docs
add_documents = not DB_DIR.exists()

if add_documents:
    documents = []
    ids = []
    for i, path in enumerate(sorted(LEGAL_DIR.glob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = text.splitlines()[0].replace("Title:", "").strip() if "Title:" in text else path.stem
        doc = Document(
            page_content=text,
            metadata={"source": str(path), "title": title}
        )
        ids.append(str(i))
        documents.append(doc)

vector_store = Chroma(
    collection_name="legal_assistant_docs",
    persist_directory=str(DB_DIR),
    embedding_function=embeddings
)

if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

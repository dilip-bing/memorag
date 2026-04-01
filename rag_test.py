"""
Quick RAG pipeline test
- Embeds a few sample documents into ChromaDB
- Asks a question and retrieves the answer
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Config ---
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

# --- Setup ---
print("Loading models...")
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)

# --- Sample documents ---
docs = [
    Document(text="Dilip is a software developer based in the US. He owns an Acer Nitro laptop with an RTX 3070 Ti GPU."),
    Document(text="The RTX 3070 Ti has 8GB of VRAM and is good for running local AI models up to 9 billion parameters."),
    Document(text="LlamaIndex is a framework for building RAG applications. It supports Ollama for local LLM inference."),
    Document(text="ChromaDB is a vector database that stores embeddings and allows semantic search over documents."),
    Document(text="bge-m3 is an embedding model by BAAI that supports 8192 token context and achieves top MTEB scores."),
]

# --- Build index ---
print("Building index (this embeds the documents)...")
index = VectorStoreIndex.from_documents(docs)

# --- Query ---
query_engine = index.as_query_engine(similarity_top_k=3)

print("\n--- RAG Pipeline Test ---\n")
questions = [
    "What GPU does Dilip have?",
    "What is ChromaDB used for?",
    "Which embedding model has the best retrieval quality?",
]

for q in questions:
    print(f"Q: {q}")
    response = query_engine.query(q)
    print(f"A: {response}\n")

print("Test complete!")

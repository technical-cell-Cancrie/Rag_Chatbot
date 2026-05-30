from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

from groq import Groq

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import Settings

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb
import os

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FREE local embeddings
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"
)

# Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# Load ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

chroma_collection = chroma_client.get_or_create_collection(
    "cancrie_rag"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# Load vector index
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store
)

# Retriever
retriever = index.as_retriever(
    similarity_top_k=10
)

@app.get("/")
def home():
    return {
        "message": "Cancrie RAG chatbot running"
    }

@app.get("/chat")
def chat(q: str):

    # retrieve relevant chunks
    nodes = retriever.retrieve(q)

    context = "\n\n".join([
        node.text for node in nodes
    ])

    # anti hallucination prompt
    prompt = f"""
You are Cancrie's technical AI assistant.

Answer ONLY using the provided context.

RULES:
- Do NOT invent information
- Do NOT make assumptions
- If information is not present in the context, say:
  "I could not find that information in the knowledge base."
- Use precise technical terminology from the context
- Keep answers factual, concise, and grounded
- Put each bullet point on a new line
- Leave a blank line between sections
- Format answers with proper spacing for readability
- Never combine multiple bullet points in a single line
- Format answers in a clean, readable way
- Use bullet points when appropriate
- Highlight important technical values clearly
- Explain findings naturally instead of copying chunks directly
- Maximum 150 words

CONTEXT:
{context}

QUESTION:
{q}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "response": answer
    }
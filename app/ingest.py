from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import Settings

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import chromadb

# FREE local embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# load documents
documents = SimpleDirectoryReader("data").load_data()

# chroma db setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = chroma_client.get_or_create_collection(
    "cancrie_rag"
)

vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# create vector index
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("RAG ingestion completed.")
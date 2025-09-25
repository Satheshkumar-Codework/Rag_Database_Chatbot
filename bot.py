# pip install streamlit langchain-openai llama-index python-dotenv chromadb llama-index-retrievers-bm25
 
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
 
# === LlamaIndex modules ===
from llama_index.core import (
    VectorStoreIndex, StorageContext, load_index_from_storage,
    Document as LlamaDocument
)
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
 
# === Load environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
 
# === Config ===
PDF_PATHS = [
    "AI-Services.pdf",
    "ChatDoc.pdf"
]
PERSIST_DIR = "vector_store"
COMPANY_EMAIL = "sales@codework.ai"
COMPANY_PHONE = "+91 75989 81500"
 
# === 1. Initialize vector index with embedding model ===
@st.cache_resource
def load_or_build_index():
    # Check if all PDF files exist
    missing_files = [path for path in PDF_PATHS if not os.path.exists(path)]
    if missing_files:
        st.error(f"PDF files not found: {', '.join(missing_files)}")
        st.stop()

    # Use faster embedding model with optimized settings
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        embed_batch_size=100,  # Process embeddings in batches
        timeout=10  # Reduce timeout for faster failures
    )

    if not os.path.exists(PERSIST_DIR):
        loader = PDFReader()
        all_docs = []
        
        # Load all PDF files
        for pdf_path in PDF_PATHS:
            raw_docs = loader.load_data(Path(pdf_path))
            all_docs.extend(raw_docs)
        
        # Combine all documents
        combined_text = "\n".join([doc.text for doc in all_docs])

        # Ultra-optimized chunking for fastest retrieval
        splitter = SentenceSplitter(chunk_size=1500, chunk_overlap=200)
        nodes = splitter.get_nodes_from_documents([LlamaDocument(text=combined_text)])

        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model
        )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(
            storage_context=storage_context,
            embed_model=embed_model
        )

    return index
 
    
 
# === 2. Hybrid Retriever class ===
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever, top_k=1):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
 
    def _retrieve(self, query, **kwargs):
        # Ultra-fast: Use only vector search for maximum speed
        vector_nodes = self.vector_retriever.retrieve(query)
        return vector_nodes[:self.top_k]
 
# === 3. LangChain-compatible wrapper ===
def get_langchain_hybrid_retriever(index):
    nodes = list(index.storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=1)
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=1)

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)
 
    def retrieve(query: str) -> list[Document]:
        results = hybrid.retrieve(query)
        return [Document(page_content=n.node.get_content(), metadata=n.node.metadata or {}) for n in results]
 
    return retrieve
 
# === 4. RAG Chains ===
def get_context_retriever_chain(retriever_func):
    # Skip LLM-based query generation to eliminate API call
    def direct_retrieve(inputs):
        return retriever_func(inputs["input"])
    
    return RunnableLambda(direct_retrieve)

def get_conversational_rag_chain(retriever_chain):
    # Ultra-fast model for response generation
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", 
        temperature=0.1, 
        max_tokens=100,  # Reduced for faster generation
        timeout=10,  # Reduced timeout
        request_timeout=10
    )
    fallback = "We don't have that information. Please contact support at sales@codework.ai or feel free to reach us at +91 75989 81500"
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are CodeWork's assistant. Respond as 'we at CodeWork' when discussing services. Use context only. Be brief. 1-2 sentences max. "
         f"Unknown: '{fallback}'\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(llm, prompt))
 
 
st.set_page_config(page_title="CodeWork Chatbot", page_icon="🤖")
st.title("🤖 CodeWork Support Bot")
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! I'm CodeWork's assistant. How can I help?")]

# Add response cache for common queries
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}

# Add query preprocessing for faster responses
def preprocess_query(query):
    """Preprocess query to improve retrieval speed"""
    # Remove common words and normalize
    query = query.lower().strip()
    # Keep only essential words (remove articles, prepositions)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    words = [word for word in query.split() if word not in stop_words and len(word) > 2]
    return ' '.join(words) if words else query
 
try:
    index = load_or_build_index()
    retriever_func = get_langchain_hybrid_retriever(index)
    retriever_chain = get_context_retriever_chain(retriever_func)
    rag_chain = get_conversational_rag_chain(retriever_chain)
except Exception as e:
    st.error(f"Initialization failed: {e}")
    st.stop()
 
user_query = st.chat_input("Type your message here...")
if user_query:
    try:
        # Check cache first for instant responses
        query_key = user_query.lower().strip()
        if query_key in st.session_state.response_cache:
            response = st.session_state.response_cache[query_key]
        else:
            # Show loading indicator for perceived faster response
            with st.spinner("Thinking..."):
                try:
                    # Preprocess query for faster retrieval
                    processed_query = preprocess_query(user_query)
                    response = rag_chain.invoke({
                        "chat_history": st.session_state.chat_history,
                        "input": processed_query
                    })['answer']
                except Exception as e:
                    # Fallback for timeout or API errors
                    response = "I'm experiencing high load. Please try again in a moment or contact support at sales@codework.ai"
            # Cache the response for future use
            st.session_state.response_cache[query_key] = response
        
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    except Exception as e:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=f"Error: {e}"))
 
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "User"
    with st.chat_message(role):
        st.write(message.content)
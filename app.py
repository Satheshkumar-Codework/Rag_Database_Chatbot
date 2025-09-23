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
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.error("OPENAI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# === Config ===
PDF_PATH = "AI-Services.pdf"
PERSIST_DIR = "vector_store"
COMPANY_EMAIL = "sales@codework.ai"
COMPANY_PHONE = "+91 75989 81500"

# === 1. Initialize vector index with embedding model ===
@st.cache_resource
def load_or_build_index():
    if not os.path.exists(PDF_PATH):
       #st.error("PDF file not found.")
        st.stop()

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    if not os.path.exists(PERSIST_DIR):
        loader = PDFReader()
        raw_docs = loader.load_data(Path(PDF_PATH))
        combined_text = "\n".join([doc.text for doc in raw_docs])

        splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
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
    def __init__(self, vector_retriever, bm25_retriever, top_k=5):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query)
        vector_nodes = self.vector_retriever.retrieve(query)
        combined = bm25_nodes + vector_nodes

        combined_sorted = sorted(combined, key=lambda x: x.score or 0, reverse=True)
        final_nodes, seen = [], set()

        for n in combined_sorted:
            if n.node.node_id not in seen:
                final_nodes.append(n)
                seen.add(n.node.node_id)
            if len(final_nodes) >= self.top_k:
                break
        return final_nodes

# === 3. LangChain-compatible wrapper ===
def get_langchain_hybrid_retriever(index):
    nodes = list(index.storage_context.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)

    hybrid = HybridRetriever(vector_retriever, bm25_retriever)

    def retrieve(query: str) -> list[Document]:
        results = hybrid.retrieve(query)
        return [Document(page_content=n.node.get_content(), metadata=n.node.metadata or {}) for n in results]

    return retrieve

# === 4. RAG Chains ===
def get_context_retriever_chain(retriever_func):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query based on this conversation."),
    ])
    return create_history_aware_retriever(llm, RunnableLambda(retriever_func), prompt)

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1, max_tokens=250)
    fallback = "I don’t have that information. Please contact support."
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are CodeWork's support assistant. Use only the provided context. "
         "Be concise and meaningful. Answer in 3–4 sentences maximum. "
         "If listing multiple items, use short bullet points. "
         f"If you don’t know the answer, say: '{fallback}'\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    return create_retrieval_chain(retriever_chain, create_stuff_documents_chain(llm, prompt))

# === 5. Streamlit App ===
st.set_page_config(page_title="CodeWork Chatbot", page_icon="🤖")
st.title("🤖 CodeWork Support Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! I'm CodeWork's assistant. How can I help?")]

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
        response = rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })['answer']
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    except Exception as e:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=f"Error: {e}"))

for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "User"
    with st.chat_message(role):
        st.write(message.content)

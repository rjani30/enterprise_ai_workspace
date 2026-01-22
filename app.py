import streamlit as st
import requests
import time
import os
import sys
import subprocess
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. MODEL CONFIGURATION & DESCRIPTIONS ---
# Restoring all models to the guide
MODEL_GUIDE = {
    "deepseek-coder-v2:16b": "🏗️ **Heavyweight Coder**: Best for complex SAS/SAP to Snowflake migrations. High accuracy.",
    "deepseek-coder:6.7b": "⚡ **Fast Coder**: Ideal for RAG-assisted code lookups and mid-sized logic.",
    "sqlcoder:7b": "🔍 **SQL Specialist**: Specifically tuned for Snowflake SQL generation and schema optimization.",
    "llama3.2:3b": "💬 **General Assistant**: Excellent for general technical chat and quick summaries.",
    "deepseek-r1:7b": "🧠 **Reasoning Expert**: Best for troubleshooting logic errors or explaining complex steps."
}

st.set_page_config(page_title="Enterprise AI Workspace", page_icon="🤖", layout="wide")

# Initialize Session State for Chat, Training, and RAG
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- 2. SIDEBAR: SETTINGS, RAG, & MONITOR ---
with st.sidebar:
    st.title("Settings & Status")
    
    # Dynamic Model Selector
    st.subheader("Model Selection")
    try:
        response = requests.get("http://ollama:11434/api/tags")
        if response.status_code == 200:
            available_models = [m['name'] for m in response.json().get('models', [])]
        else:
            available_models = list(MODEL_GUIDE.keys())
    except Exception:
        available_models = list(MODEL_GUIDE.keys())
        st.error("Connecting to Ollama engine...")

    selected_model = st.selectbox("Choose an Active Model", options=available_models)
    st.info(MODEL_GUIDE.get(selected_model, "🌐 **General Purpose**: Standard AI model."))

    st.divider()

    # RAG: DOCUMENT UPLOADER & INDEXING
    st.subheader("📁 RAG Data Ingestion")
    # Persistent Status Indicator
    if st.session_state.vector_db:
        st.success("✅ RAG Index Active")
    else:
        st.info("ℹ️ No Document Indexed")
    uploaded_file = st.file_uploader("Upload script/doc (PDF, TXT, SAS, PY)", type=["pdf", "txt", "py", "sas", "sql"])
    
    if uploaded_file and st.button("🚀 Index for RAG"):
        with st.spinner("Processing document into searchable chunks..."):
            try:
                # Extract Text
                if uploaded_file.type == "application/pdf":
                    reader = PdfReader(uploaded_file)
                    raw_text = "".join([p.extract_text() for p in reader.pages])
                else:
                    raw_text = uploaded_file.getvalue().decode("utf-8")
                
                # Smart Chunking for RAG
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_text(raw_text)
                
                # Local Vector Store (FAISS)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)
                st.success(f"Indexed {len(chunks)} chunks successfully!")
            except Exception as e:
                st.error(f"Indexing Error: {e}")

    st.divider()

    # Memory Controls
    max_history = st.slider("Max Context History", 1, 20, 5)
    if st.button("🗑️ Clear All History"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.rerun()

    st.divider()

    # Training Monitor
    if 'training_proc' in st.session_state:
        st.subheader("Training Monitor")
        ret_code = st.session_state.training_proc.poll()
        if ret_code is None: st.warning("⏳ Training in Progress...")
        elif ret_code == 0: st.success("✅ Training Complete!")
        else: st.error(f"❌ Training Failed (Code: {ret_code})")

# --- 3. MAIN INTERFACE ---
st.title("Enterprise AI Workspace")
tab1, tab2 = st.tabs(["💬 RAG Chat Assistant", "🚀 Expert Trainer"])

# --- TAB 1: RAG CHAT ASSISTANT ---
with tab1:
    st.header(f"Chatting with {selected_model}")
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question about the code or your document...")

    if user_input:
        # Show User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # RAG Retrieval Logic: Find relevant chunks
        context = ""
        if st.session_state.vector_db:
            docs = st.session_state.vector_db.similarity_search(user_input, k=4)
            context = "\n\n".join([d.page_content for d in docs])
            st.caption(f"🔍 Found {len(docs)} relevant document segments.")

        # Construct Final Prompt with Context and History
        history_window = st.session_state.chat_history[-(max_history * 2):]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_window])
        
        full_prompt = f"DOCUMENT CONTEXT:\n{context}\n\nCHAT HISTORY:\n{history_text}\nassistant:"
        
        placeholder = st.empty()
        start_time = time.perf_counter()
        placeholder.image("https://i.gifer.com/ZZ5H.gif", caption="RAG Engine Searching & Thinking...", width=100)
        
        try:
            # Query the AI Engine
            resp = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": selected_model, 
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"num_ctx": 8192} # Sufficient for RAG results
                },
                timeout=300
            )
            
            execution_time = time.perf_counter() - start_time
            
            if resp.status_code == 200:
                placeholder.empty()
                ai_response = resp.json().get("response")
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.caption(f"⏱️ Response generated in **{execution_time:.2f} seconds**")
            else:
                placeholder.empty()
                st.error(f"AI Engine Busy: {resp.status_code}")
                    
        except Exception as e:
            placeholder.empty()
            st.error(f"Connection Error: {e}")

# --- TAB 2: EXPERT TRAINER ---
with tab2:
    st.header("Custom Fine-Tuning")
    st.markdown("Upload your `.jsonl` dataset to start local training.")
    dataset = st.file_uploader("Upload .jsonl Dataset", type=["jsonl"])

    is_training = 'training_proc' in st.session_state and st.session_state.training_proc.poll() is None
    
    if st.button("🚀 Start Local Fine-Tuning", disabled=is_training):
        if dataset:
            with open("train_data.jsonl", "wb") as f:
                f.write(dataset.getbuffer())
            try:
                st.session_state.training_proc = subprocess.Popen([sys.executable, "finetune.py"])
                st.success("Training started! Monitor progress in the sidebar.")
                st.rerun()
            except Exception as e:
                st.error(f"Launch Error: {e}")
        else:
            st.error("Please upload a dataset first.")
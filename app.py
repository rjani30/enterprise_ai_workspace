import streamlit as st
import subprocess
import requests
import sys
import os
import time
from pypdf import PdfReader 

# --- 1. MODEL GUIDE & CONFIG ---
MODEL_GUIDE = {
    "deepseek-coder-v2:16b": "🏗️ **Heavyweight Coder**: Best for complex SAS/SAP to Snowflake migrations. High accuracy but uses more RAM.",
    "deepseek-coder:6.7b": "⚡ **Fast Coder**: Ideal for routine SQL debugging and mid-sized logic scripts.",
    "llama3.2:3b": "💬 **General Assistant**: Best for general technical chat and quick summaries.",
    "sqlcoder:7b": "🔍 **SQL Specialist**: Specifically tuned for advanced Snowflake SQL optimization.",
    "deepseek-r1:7b": "🧠 **Reasoning Expert**: Great for troubleshooting complex logic or step-by-step explanations."
}

st.set_page_config(page_title="Enterprise AI Workspace", page_icon="🤖", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "current_chunk_index" not in st.session_state:
    st.session_state.current_chunk_index = 0

# --- 3. HELPER FUNCTION: TEXT CHUNKING ---
def chunk_text(text, size=2000):
    """Breaks large text into manageable pieces for processing."""
    return [text[i : i + size] for i in range(0, len(text), size)]

# --- 4. SIDEBAR: SETTINGS & MODEL SELECTION ---
with st.sidebar:
    st.title("Settings & Status")
    
    st.subheader("Model Selection")
    try:
        # Dynamic connection to Ollama container
        response = requests.get("http://ollama:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            available_models = [m['name'] for m in models_data]
        else:
            available_models = ["deepseek-coder-v2:16b"]
    except Exception:
        available_models = ["deepseek-coder-v2:16b"]
        st.error("Connecting to AI engine...")

    selected_model = st.selectbox("Choose an Active Model", options=available_models)
    st.info(MODEL_GUIDE.get(selected_model, "🌐 **General Purpose**: Standard AI model."))

    # --- DOCUMENT UPLOADER & CHUNKING ---
    st.divider()
    st.subheader("📁 Document Management")
    uploaded_file = st.file_uploader("Upload large script/doc", type=["txt", "py", "sas", "pdf", "sql"])

    if uploaded_file:
        raw_text = ""
        try:
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                raw_text = "".join([page.extract_text() for page in reader.pages])
            else:
                raw_text = uploaded_file.getvalue().decode("utf-8")
            
            # Update chunks
            st.session_state.chunks = chunk_text(raw_text)
            st.success(f"File loaded: {len(st.session_state.chunks)} chunks created.")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    # Chunk Navigation Control
    if st.session_state.chunks:
        st.session_state.current_chunk_index = st.number_input(
            "Target Chunk Index", 
            min_value=0, 
            max_value=len(st.session_state.chunks)-1, 
            value=st.session_state.current_chunk_index
        )

    # --- MEMORY TUNING ---
    st.divider()
    max_history = st.slider("Max Chat Memory (Turns)", 1, 20, 5, help="Number of previous messages for context.")
    
    if st.button("🗑️ Reset All Sessions"):
        st.session_state.chat_history = []
        st.session_state.chunks = []
        st.rerun()

    st.divider()
    # Training Status Monitor
    if 'training_proc' in st.session_state:
        st.subheader("Training Monitor")
        ret_code = st.session_state.training_proc.poll()
        if ret_code is None: st.warning("⏳ Training in Progress...")
        elif ret_code == 0: st.success("✅ Training Complete!")
        else: st.error(f"❌ Failed (Code: {ret_code})")

# --- 5. MAIN INTERFACE ---
st.title("Enterprise AI Workspace")
tab1, tab2 = st.tabs(["💬 General Assistant", "🚀 Expert Trainer"])

with tab1:
    st.header(f"Chat Engine: {selected_model}")
    
    # Optional Preview of active chunk
    if st.session_state.chunks:
        with st.expander("📄 View Active Chunk Content"):
            st.code(st.session_state.chunks[st.session_state.current_chunk_index])

    # Display Conversation History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- ANALYSIS MODE SELECTION ---
    use_full_doc = st.checkbox("🔄 Process ALL chunks together (Full Doc Analysis)", value=False)
    
    user_input = st.chat_input("Ask a question about the document or code...")

    if user_input:
        # 1. Update UI with user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # 2. Construct Prompt Context
        if use_full_doc and st.session_state.chunks:
            # Join all text to process the 11+ page doc at once
            active_context = "\n\n".join(st.session_state.chunks)
            display_label = "Analyzing whole document..."
        elif st.session_state.chunks:
            # Only use the specific selected chunk
            active_context = st.session_state.chunks[st.session_state.current_chunk_index]
            display_label = f"Analyzing chunk {st.session_state.current_chunk_index}..."
        else:
            active_context = ""
            display_label = "Thinking..."

        # Format history window
        history_window = st.session_state.chat_history[-(max_history * 2):]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_window])
        
        # Final combined prompt
        full_prompt = f"CONTEXT:\n{active_context}\n\nCHAT HISTORY:\n{history_text}\nassistant:"
        
        # 3. Request Generation with Progress UI
        placeholder = st.empty()
        start_time = time.perf_counter()
        placeholder.image("https://i.gifer.com/ZZ5H.gif", caption=display_label, width=100)
        
        try:
            # --- CRITICAL: EXPAND CONTEXT WINDOW & TIMEOUT ---
            resp = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": selected_model, 
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 32768,  # Increased memory for large docs
                        "temperature": 0.1
                    }
                },
                timeout=600 # 10-minute limit for full-doc analysis
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
                st.error(f"AI Engine Error ({resp.status_code})")
        except Exception as e:
            placeholder.empty()
            st.error(f"Connection Error: {e}")

# --- TAB 2: TRAINING (Logic remains separate for background safety) ---
with tab2:
    st.header("Local Model Fine-Tuning")
    # ... existing training logic ...
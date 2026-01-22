import streamlit as st
import subprocess
import requests
import sys
import os
import time

# --- 1. MODEL GUIDE & CONFIG ---
MODEL_GUIDE = {
    "deepseek-coder-v2:16b": "🏗️ **Heavyweight Coder**: Best for complex migrations. High accuracy.",
    "deepseek-coder:6.7b": "⚡ **Fast Coder**: Good for mid-sized logic and SQL debugging.",
    "llama3.2:3b": "💬 **General Assistant**: Best for general questions and chat memory.",
    "sqlcoder:7b": "🔍 **SQL Specialist**: Specifically tuned for Snowflake SQL."
}

st.set_page_config(page_title="Enterprise AI Workspace", page_icon="🤖", layout="wide")

# Initialize Chat History if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 2. SIDEBAR ---
with st.sidebar:
    st.title("Settings & Status")
    
    st.subheader("Model Selection")
    try:
        response = requests.get("http://ollama:11434/api/tags")
        available_models = [m['name'] for m in response.json().get('models', [])] if response.status_code == 200 else ["deepseek-coder-v2:16b"]
    except:
        available_models = ["deepseek-coder-v2:16b"]
    
    selected_model = st.selectbox("Choose an Active Model", options=available_models)
    st.info(MODEL_GUIDE.get(selected_model, "🌐 **General Purpose**"))

    # --- NEW: MAX HISTORY SLIDER ---
    st.divider()
    st.subheader("Memory Tuning")
    max_history = st.slider(
        "Max Context History", 
        min_value=1, 
        max_value=20, 
        value=5, 
        help="How many previous messages the AI should remember. Higher values use more CPU/RAM."
    )
    
    if st.button("🗑️ Clear Chat Context"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    
    # Training Monitor
    if 'training_proc' in st.session_state:
        st.subheader("Training Monitor")
        ret_code = st.session_state.training_proc.poll()
        if ret_code is None: st.warning("⏳ Training in Progress...")
        elif ret_code == 0: st.success("✅ Training Complete!")
        else: st.error(f"❌ Failed (Code: {ret_code})")

# --- 3. MAIN INTERFACE ---
st.title("Enterprise AI Workspace")
tab1, tab2 = st.tabs(["💬 General Assistant", "🚀 Expert Trainer"])

with tab1:
    st.header(f"Conversing with {selected_model}")
    
    # Display previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a follow-up or provide new code...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # --- LOGIC: SLICE HISTORY BASED ON SLIDER ---
        # We take the last 'max_history * 2' entries to account for User + AI pairs
        context_window = st.session_state.chat_history[-(max_history * 2):]
        full_context = "\n".join([f"{m['role']}: {m['content']}" for m in context_window])
        
        placeholder = st.empty()
        start_time = time.perf_counter()
        loading_gif = "https://i.gifer.com/ZZ5H.gif" 
        placeholder.image(loading_gif, caption=f"Thinking with last {max_history} exchanges...", width=100)
        
        try:
            resp = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": selected_model, 
                    "prompt": full_context,
                    "stream": False
                },
                timeout=300
            )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            if resp.status_code == 200:
                placeholder.empty()
                ai_response = resp.json().get("response")
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                st.caption(f"⏱️ Context-aware response in **{execution_time:.2f} seconds**")
            else:
                placeholder.empty()
                st.error(f"Engine Error: {resp.status_code}")
                    
        except Exception as e:
            placeholder.empty()
            st.error(f"Connection Error: {e}")
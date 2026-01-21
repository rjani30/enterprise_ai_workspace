import streamlit as st
import subprocess
import requests
import sys
import os
import time

# --- 1. MODEL GUIDE DEFINITION ---
# Mapping model names to their specific expertises
MODEL_GUIDE = {
    "deepseek-coder-v2:16b": "🏗️ **Heavyweight Coder**: Best for complex SAS/SAP to Snowflake migrations. High accuracy but slower.",
    "deepseek-coder:6.7b": "⚡ **Fast Coder**: Good for mid-sized ETL logic and routine SQL debugging. Balanced speed.",
    "llama3.2:3b": "💬 **General Assistant**: Excellent for general technical questions, documentation, and simple chat.",
    "sqlcoder:7b": "🔍 **SQL Specialist**: Specifically tuned for advanced Snowflake SQL generation and schema optimization.",
    "deepseek-r1:7b": "🧠 **Reasoning Expert**: Best for troubleshooting complex logic errors or explaining step-by-step migrations."
}

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise AI Workspace", page_icon="🤖", layout="wide")

# --- 3. SIDEBAR: DYNAMIC SELECTOR & DESCRIPTIONS ---
with st.sidebar:
    st.title("Settings & Status")
    
    st.subheader("Model Selection")
    try:
        response = requests.get("http://ollama:11434/api/tags")
        if response.status_code == 200:
            models_data = response.json().get('models', [])
            available_models = [m['name'] for m in models_data]
        else:
            available_models = ["deepseek-coder-v2:16b"]
    except Exception:
        available_models = ["deepseek-coder-v2:16b"]
        st.error("Connecting to Ollama engine...")

    selected_model = st.selectbox("Choose an Active Model", options=available_models)

    # Display Description based on Selection
    description = MODEL_GUIDE.get(selected_model, "🌐 **General Purpose**: A standard model for various AI tasks.")
    st.info(description)

    st.divider()

    # Training Monitor
    st.subheader("Training Monitor")
    if 'training_proc' in st.session_state:
        ret_code = st.session_state.training_proc.poll()
        if ret_code is None:
            st.warning("⏳ Training in Progress...")
        elif ret_code == 0:
            st.success("✅ Training Complete!")
            if st.button("Clear Notification"):
                del st.session_state.training_proc
                st.rerun()
        else:
            st.error(f"❌ Training Failed (Code: {ret_code})")
    else:
        st.write("No active tasks.")

# --- 4. MAIN INTERFACE ---
st.title("Enterprise AI Workspace")
tab1, tab2 = st.tabs(["💬 General Assistant", "🚀 Expert Trainer"])

with tab1:
    st.header(f"Working with {selected_model}")
    user_prompt = st.text_area("Input Area:", height=300, placeholder="Paste your SAS/SAP code here...")

    if st.button("Send to AI Engine"):
        if user_prompt.strip():
            with st.spinner("Processing..."):
                try:
                    resp = requests.post(
                        "http://ollama:11434/api/generate",
                        json={"model": selected_model, "prompt": user_prompt, "stream": False},
                        timeout=120
                    )
                    if resp.status_code == 200:
                        st.subheader("AI Response:")
                        st.markdown(resp.json().get("response"))
                except Exception as e:
                    st.error(f"Connection Error: {e}")

with tab2:
    st.header("Custom Fine-Tuning")
    dataset = st.file_uploader("Upload .jsonl Dataset", type=["jsonl"])
    is_training = 'training_proc' in st.session_state and st.session_state.training_proc.poll() is None
    
    if st.button("🚀 Start Local Fine-Tuning", disabled=is_training):
        if dataset is not None:
            with open("train_data.jsonl", "wb") as f:
                f.write(dataset.getbuffer())
            st.session_state.training_proc = subprocess.Popen([sys.executable, "finetune.py"])
            st.rerun()
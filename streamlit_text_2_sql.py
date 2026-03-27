import streamlit as st
import yaml
import requests
from langchain_community.utilities import SQLDatabase
import re

# 1. Database Connection (pointing to the container name)
db_uri = "postgresql://user:password@smart_meter_db:5432/smart_meter_db"

try:
    db = SQLDatabase.from_uri(db_uri)
    st.sidebar.success("✅ Connected to Postgres")
except Exception as e:
    st.sidebar.error(f"❌ Connection Failed: {e}")

# 2. Load Semantic Layer
try:
    with open("semantic_layer.yaml", "r") as f:
        semantic_context = yaml.safe_load(f)
except FileNotFoundError:
    st.sidebar.warning("⚠️ semantic_layer.yaml not found.")
    semantic_context = "No additional semantic context provided."

st.title("⚡ Smart Meter AI SQL Assistant")

question = st.text_input("Ask a question about your smart meter data:")

if question:
    # 3. Prompt Construction
    # We explicitly tell the model not to use Markdown formatting
    prompt = f"""
    You are a SQL expert. 
    Semantic Layer: {semantic_context}
    Table Info: {db.get_table_info()}
    User Question: {question}
    
    Return ONLY the raw SQL query. Do not include markdown code blocks, 
    backticks, or any conversational text.
    """
    
    with st.spinner("Generating SQL via DeepSeek..."):
        try:
            resp = requests.post(
                "http://ollama-engine:11434/api/generate", 
                json={"model": "deepseek-coder-v2:16b", "prompt": prompt, "stream": False},
                timeout=30
            )
            raw_response = resp.json().get("response").strip()
            
            # --- CLEANING LOGIC ---
            # Remove Markdown code blocks (e.g., ```sql ... ```)
            sql_query = re.sub(r"```sql|```", "", raw_response, flags=re.IGNORECASE).strip()
            
            # Remove common LLM prefixes if they exist
            sql_query = re.sub(r"^(sql|query):\s*", "", sql_query, flags=re.IGNORECASE).strip()
            # ----------------------

            st.subheader("Generated SQL:")
            st.code(sql_query, language="sql")
            
            # 4. Execute and Display Result
            if sql_query:
                result = db.run(sql_query)
                st.write("### Query Results:")
                # result is often a string representation of a list of tuples from LangChain
                st.write(result)
            else:
                st.error("The model returned an empty response.")
                
        except Exception as e:
            st.error(f"Error processing request: {e}")
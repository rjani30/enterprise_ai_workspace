from mcp.server.fastmcp import FastMCP
import psycopg2
import yaml
import requests

mcp = FastMCP("smart-meter-ai")

# Load semantic layer
with open("semantic_layer.yaml") as f:
    semantic = yaml.safe_load(f)

DB_CONN = "postgresql://user:password@localhost:5432/smart_meter_db"

def generate_sql(question):

    prompt = f"""
You are a SQL expert.

Semantic Layer:
{semantic}

User Question:
{question}

Return ONLY SQL.
"""

    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-coder-v2:16b",
            "prompt": prompt,
            "stream": False
        }
    )

    return resp.json()["response"]


@mcp.tool()
def text_to_sql(question: str) -> str:
    """Convert natural language question into SQL."""
    return generate_sql(question)


@mcp.tool()
def run_sql(sql: str) -> str:
    """Execute SQL against smart meter database."""

    conn = psycopg2.connect(DB_CONN)
    cur = conn.cursor()

    cur.execute(sql)
    rows = cur.fetchall()

    conn.close()

    return str(rows)


if __name__ == "__main__":
    mcp.run()
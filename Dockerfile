# 1. Force amd64 architecture for stable pre-compiled wheels on Mac Silicon
FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

# 2. Install critical system dependencies for building Python extensions
# swig: Required for faiss-cpu
# libpq-dev & gcc: Required for Postgres/psycopg2
# python3-dev: Essential for building complex C-extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    swig \
    python3-dev \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 3. Upgrade pip and install core computation libraries first
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Unsloth (as per your previous setup requirements)
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 5. Copy and install remaining requirements (langchain-community, faiss-cpu, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your app code
COPY . .

# 7. Expose Streamlit's default port
EXPOSE 8501

# 8. Command to run the Text-to-SQL application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
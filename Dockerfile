# Upgrade to 3.10 to support Triton 3.0+ and bitsandbytes 0.49+
FROM python:3.10-slim

WORKDIR /app

# 1. Install system dependencies (build-essential helps with aarch64 compilations)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# 3. Install Torch CPU first to avoid xformers dependency conflicts
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install Unsloth from GitHub
RUN pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 5. Copy and install remaining requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
To get your Enterprise AI Workspace up and running again on your Mac, you just need to follow these four primary steps to ensure the environment, dependencies, and containers are synchronized.
1. Synchronize Your Project Files
Ensure your project folder (enterprise_ai_workspace) contains the latest versions of the files we built:
app.py: The main Streamlit script with RAG and multi-model support.
requirements.txt: Should include streamlit, requests, pypdf, langchain, langchain-community, langchain-text-splitters, sentence-transformers, and faiss-cpu.
docker-compose.yml: Defines the sas-converter-ui and ollama-engine services.
Dockerfile: Instructions for building the UI container.
2. Rebuild and Start Containers
Since we added new libraries (like LangChain), you must rebuild the Docker image to install them into the container's environment. Run this in your terminal:

Bash


cd path/to/enterprise_ai_workspace
docker compose up --build -d


--build: Forces Docker to re-read your requirements.txt and install the new RAG dependencies.
-d: Runs the app in the background so you can close the terminal window.
3. Ensure Models are Loaded
The ollama-engine needs the actual AI model files to perform the work. If you haven't pulled them yet (or if they were cleared), run these commands:

Bash


docker exec -it ollama-engine ollama pull deepseek-coder-v2:16b
docker exec -it ollama-engine ollama pull llama3.2:3b


(You can repeat this for sqlcoder:7b or any other models in your guide).
4. Access the Application
Once the containers are healthy, open your web browser and go to:
http://localhost:8501
Quick Troubleshooting Check
Memory: Ensure Docker Desktop has at least 16GB–24GB of RAM allocated in its settings, or the 16B model will fail to load.
CPU: If the app is sluggish, check that your Mac isn't thermal throttling (close other heavy apps like Chrome or Teams).
Would you like me to provide a "Startup Health Check" script you can run to verify that all containers and libraries are working correctly in one go?

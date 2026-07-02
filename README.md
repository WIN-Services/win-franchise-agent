# Franchise Chatbot System

A production-ready Franchise Chatbot System designed to support Strategic Partner (SP) acquisition and onboarding. It acts as a Franchise Knowledge Assistant with a strict 5-layer architecture.

## Project Architecture

1. **Data Layer** (`rag/`): Web scraping & document ingestion, chunking, and FAISS vector storage.
2. **Tooling Layer** (`app/tools/`): Internal APIs isolating data from the agent.
3. **Agent Layer** (`app/agents/`): LangChain MVP Agent acting as Franchise Assistant.
4. **Orchestration Layer** (`app/router.py`): Routing input to the right tools and agents.
5. **Interface Layer** (`app/main.py`): FastAPI exposing the `/chat` endpoint.

## Prerequisites
- Python 3.9+
- OpenAI API Key

## Setup Steps

1. **Clone/Navigate to the repository**
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Environment Variables**
   Update the `.env` file in the project root with your `OPENAI_API_KEY` and any other specific configurations.

5. **Run the API**
   ```bash
   uvicorn app.main:app --reload --port 9050
   ```

## Testing the API
You can test the API by visiting `http://localhost:9050/docs` in your browser, or using curl:

```bash
curl -X POST http://localhost:9050/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the franchise cost?"}'
```

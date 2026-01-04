# SLAA AI Sponsor - Recovery Companion

## Overview
This is an Arabic-language AI chatbot application that serves as a recovery sponsor. It uses LangChain with Groq LLM to provide guidance based on PDF documents in the library folder.

## Tech Stack
- Python 3.11
- Streamlit (frontend/UI)
- LangChain 0.1.20 (LLM orchestration)
- Groq (LLM provider - llama3-70b-8192 model)
- FastEmbed (embeddings)
- FAISS (vector store)
- PyPDF (PDF processing)

## Project Structure
- `main.py` - Main Streamlit application
- `library/` - Directory containing PDF documents for the knowledge base
- `requirements.txt` - Python dependencies

## Configuration
- The app requires a `GROQ_API_KEY` environment variable/secret
- Streamlit runs on port 5000
- The app uses memory-optimized settings (single thread, smaller chunks)

## Running the App
The app is configured to run via Streamlit on port 5000:
```
streamlit run main.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
```

## Features
- Loads and processes PDF documents from the library folder
- Creates vector embeddings for semantic search
- Provides AI-powered responses in Arabic based on the document context
- Chat interface with message history

# Codebase Onboarding Assistant (RAG Pipeline)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)

An AI-powered, framework-free Retrieval-Augmented Generation (RAG) assistant designed to solve developer onboarding pain, mitigate documentation drift, and improve codebase discoverability. Chat directly with your documentation using a local LLM—no expensive APIs or heavy orchestration frameworks (like LangChain or LlamaIndex) required.

## Table of Contents
- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Command Line Interface (CLI)](#1-command-line-interface-cli)
  - [Streamlit Web App](#2-streamlit-web-app)
  - [Evaluation Pipeline](#3-evaluation-pipeline)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Customization](#%EF%B8%8F-customization)

## Features
- **Semantic Search**: Fast and accurate vector search using `FAISS` and `SentenceTransformers`.
- **Local LLM Integration**: Uses [Ollama](https://ollama.ai/) with `llama3.2` for fully local, privacy-preserving generation.
- **Dual Interfaces**: Choose between a lightweight CLI terminal or an interactive web-based Streamlit UI.
- **Stale Documentation Detection**: Automatically warns you if retrieved documents haven't been updated in over 180 days.
- **Low-Confidence Warnings**: Flags answers generated from weakly matching sources.
- **Custom Evaluation Engine**: Built-in script to test retrieval accuracy, faithfulness, and safety against a golden dataset.

## Architecture
1. **Ingestion Layer**: Recursively loads markdown documents from the `docs/` directory.
2. **Processing Layer**: Splits text into semantic chunks and generates dense vector embeddings.
3. **Retrieval Layer**: Uses a FAISS `IndexFlatL2` vector database to perform high-speed similarity searches.
4. **Generation Layer**: Constructs prompt contexts and queries a local `llama3.2` model via Ollama. 

## Prerequisites
Before you begin, ensure you have the following installed:
- **Python 3.8+**
- **Ollama**: Required to run the local LLM.
- **Llama 3.2**: Download the specific model via Ollama:
  ```bash
  ollama run llama3.2
  ```

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-link>
   cd "Code-Base Onboarding"
   ```

2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your documentation**:
   Place all your markdown (`.md`) files inside the `docs/` folder in the project root.

## Usage

### 1. Command Line Interface (CLI)
Interact with the assistant directly from your terminal. This is great for debugging or quick queries.
```bash
python main.py
```
*Note: The first run might take slightly longer as it generates and caches embeddings into an `embeddings_data.pkl` file.*

### 2. Streamlit Web App
Launch an interactive, user-friendly browser UI.
```bash
python -m streamlit run app.py
```
This application will automatically load your cached indexing and provide a chat interface where you can view contextual sources and debug information.

### 3. Evaluation Pipeline
Test the system's accuracy and faithfulness using the built-in automated evaluation suite. Ensure you have `evaluation/golden_dataset.json` populated with test cases.
```bash
python evaluate.py
```
This script will output an `evaluation_results.csv` detailing the performance over your expected queries.

## Project Structure

```text
Code-Base Onboarding/
├── app.py                  # Streamlit web application
├── main.py                 # Interactive CLI script
├── evaluate.py             # Accuracy/faithfulness evaluation script
├── requirements.txt        # Python dependency list
├── docs/                   # Directory containing your markdown source files
├── evaluation/             # Test datasets and evaluation results
│   └── golden_dataset.json 
├── ingestion/              # Data parsing and loading logic
├── processing/             # Text chunking and embedding generation
├── retrieval/              # Vector DB, query mapping, and prompt building
└── utils/                  # Helper functions (e.g., caching embeddings)
```

## How It Works
1. **Load**: `os.walk()` fetches markdown files and their metadata (last modified time).
2. **Chunk**: Text is logically split using custom RegEx while preserving markdown headers or logical boundaries.
3. **Embed**: `SentenceTransformers` (`all-MiniLM-L6-v2` by default) converts chunks into semantic embeddings.
4. **Retrieve**: When a user asks a question, it is embedded, compared against the FAISS index (Top K=5), and retrieved.
5. **Answer**: The context is concatenated and injected alongside the user query to the local LLM to generate a grounded, hallucination-free answer.

## Customization
You can tweak model settings and heuristics depending on your needs. For example, in `main.py` and `evaluate.py`:
- `STALE_THRESHOLD_DAYS = 180`: Adjusts when the UI warns about old documentation.
- `TOP_K = 5`: Number of document chunks to fetch during retrieval.
- `MAX_CONTEXT_CHARS = 2500`: The context string length limit passed to the LLM to save tokens and latency.
- Modify the URL in `call_llm` if you run Ollama on a different port or network host.

---
*Created as an end-to-end framework-free approach to building highly capable developer tooling with AI.*

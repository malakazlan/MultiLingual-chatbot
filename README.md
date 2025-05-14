# Multilingual RAG Knowledge Base

This project implements a multilingual Retrieval-Augmented Generation (RAG) knowledge base system for document chunking, embedding, and semantic search using Pinecone as a vector store. It is designed to support finance-related documents and can be extended to other domains.

## Features
- **Text Preprocessing & Chunking:**
  - Cleans and splits raw text files into manageable chunks using fixed-length and sentence-based strategies.
- **Multilingual Embedding:**
  - Supports multiple embedding models (SBERT, LaBSE, E5) for robust semantic representation.
- **Vector Store Integration:**
  - Uses Pinecone for scalable vector storage and fast similarity search.
- **Metadata Extraction:**
  - (Optional) Extracts named entities and date information for advanced filtering.
- **Interactive Search:**
  - Command-line interface for querying the knowledge base and retrieving the most relevant document chunks.

## Usage

1. **Preprocess and Chunk Documents**
   - Place your raw `.txt` files in `multilingual_rag_kb/data/raw/`.
   - (Optional) Run the preprocessing and chunking functions to generate clean, chunked data in `data/processed/`.

2. **Embed and Upload Chunks to Pinecone**
   - Use the `push_to_pinecone()` function to embed chunks and upload them to the Pinecone vector store.
   - Configure your Pinecone API key, environment, and index name in `multilingual_rag_kb/config.py`.

3. **Run Similarity Search**
   - Use the `run_similarity_search()` function to enter a query and retrieve the top matching chunks from Pinecone.
   - Results include content previews and (optionally) extracted entities and date flags.

4. **(Optional) RAG Chat**
   - Integrate with an LLM (e.g., via Ollama) to generate answers using retrieved context chunks.

## Requirements
- Python 3.8+
- Pinecone
- NLTK, sentence-transformers, and other dependencies (see your environment setup)

## Project Structure
- `chunking/` - Chunking utilities
- `data/` - Raw and processed data
- `embeddings/` - Embedding logic
- `llm/` - LLM integration and prompt templates
- `models/` - Embedding model wrappers
- `utils/` - Text cleaning and helpers
- `vector_store/` - Pinecone and other vector store integrations

## Notes
- Entity extraction and date filtering are optional and can be enabled by uncommenting the relevant code in `app.py`.
- Make sure to set up your Pinecone account and environment variables before running the upload or search scripts.

---

For more details, see the code in `app.py` and the respective module folders. 

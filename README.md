<div align="center">

# NVIDIA NIM Q&A Generator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green)](https://python.langchain.com/)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA-NIM-76B900)](https://www.nvidia.com/en-us/ai-data-science/products/nim/)
[![License](https://img.shields.io/badge/License-GPL-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


</div>

## 📋 Overview

The NVIDIA NIM Q&A Generator is a powerful application that leverages NVIDIA's NIM (NVIDIA Inference Microservices) and LLaMa 3 70B model to generate question and answer pairs from PDF documents. This tool allows users to upload PDF files, process them using advanced NLP techniques, and generate comprehensive Q&A pairs based on the document content.

Built on a robust technical stack, the application combines LangChain's document processing capabilities, FAISS vector database for efficient similarity search, and NVIDIA's state-of-the-art AI models to deliver high-quality question and answer generation. The system utilizes semantic embeddings to understand document context and leverages NVIDIA's optimized inference infrastructure for fast, accurate responses.

## Landing Page

![Landing Page](landing.png)

## ✨ Features

- **PDF Document Processing**: Upload and process multiple PDF documents
- **Vector Database Creation**: Automatically creates embeddings and vector stores for efficient retrieval
- **Q&A Generation**: Generate multiple question and answer pairs based on document content
- **NVIDIA NIM Integration**: Utilizes NVIDIA's powerful AI models for high-quality responses
- **Interactive UI**: User-friendly Streamlit interface for easy interaction

## 🔧 Technical Architecture

- **Frontend**: Streamlit for the user interface with interactive components for document upload and Q&A generation
- **Embeddings**: NVIDIA AI Embeddings for high-dimensional vector representation of text, optimized for semantic search
- **LLM**: Meta's LLaMa 3 70B model via NVIDIA NIM (NVIDIA Inference Microservices) for state-of-the-art natural language processing
- **Vector Store**: FAISS (Facebook AI Similarity Search) for efficient similarity search and retrieval with:  
  - High-performance indexing of dense vectors
  - Fast nearest neighbor search capabilities
  - Memory-efficient storage of document embeddings
- **Document Processing**: LangChain framework components including:
  - Document loaders for PDF parsing and extraction
  - Text splitters for creating semantically meaningful chunks
  - Retrieval mechanisms for context-aware Q&A generation
  - Chain-of-thought prompting for improved response quality

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- NVIDIA API Key (starts with 'nvapi')

### Setup

1. Clone the repository:
   ```bash
   git clone
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. Start the Streamlit application:
   ```bash
   streamlit run nvidia_nim.py
   ```

2. Enter your NVIDIA API Key when prompted

3. Upload PDF documents or use the default documents in the `temp` directory

4. Click "Prepare Document Database" to process the documents

5. Enter the number of Q&A pairs you want to generate

6. View the generated Q&A pairs and the source documents

## 🔑 API Key

To use this application, you need a valid NVIDIA API Key that starts with 'nvapi'. You can obtain one by:

1. Visiting the [NVIDIA Developer Portal](https://developer.nvidia.com/)
2. Creating or logging into your NVIDIA account
3. Navigating to the API section and generating a new API key

## 📊 How It Works

1. **Document Processing**: PDF documents are loaded using LangChain's document loaders and split into manageable chunks with intelligent text splitters that preserve semantic meaning
2. **Embedding Creation**: Document chunks are converted into high-dimensional vector embeddings using NVIDIA's NIM embedding models, which capture semantic relationships between text segments
3. **Vector Database**: Embeddings are stored in a FAISS (Facebook AI Similarity Search) vector database, which provides:
   - Sub-linear time complexity for similarity search
   - Efficient indexing of high-dimensional vectors
   - Optimized memory usage for large document collections
4. **Query Processing**: LangChain's retrieval mechanisms process user queries to find the most relevant document chunks using cosine similarity in the vector space
5. **Response Generation**: The LLaMa 3 70B model, served through NVIDIA NIM, generates contextually accurate Q&A pairs based on the retrieved chunks, leveraging:
   - NVIDIA's optimized inference infrastructure
   - LangChain's chain-of-thought prompting techniques
   - Context-aware response generation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the GNU General Public License (GPL) - see the LICENSE file for details.

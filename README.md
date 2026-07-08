🚀 Sajalni AI Classifier — Intelligent RAG Email Routing System
An automated email classification and routing pipeline designed for the Tunisian Sajalni (CEIR) platform. This system processes incoming user requests, handles dialectal and multilingual complexities, and matches them with the correct administrative response templates using a production-ready Retrieval-Augmented Generation (RAG) architecture.

🌟 Key Features
Multilingual Semantic Search: Uses intfloat/multilingual-e5-base embeddings to match emails in French, English, and Modern Standard Arabic.

Localized Dialectal Anchor (Derja/Arabizi): Built-in semantic boosters to intercept local Tunisian dialect and Arabizi (e.g., nbloki, tserreqli, chkoun) to strictly enforce business compliance rules.

Robust Text Chunking: Implements token-aware text chunking via Chonkie to prevent context overflows from signature blocks or long email histories.

Hybrid Decision Engine: Uses Qdrant for vector similarity search paired with a local Ollama (Qwen) LLM instance to double-validate categorization decisions.

Production-Ready Dockerization: Orchestrated with Docker Compose for seamless replication across development and production environments.

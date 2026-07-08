# Sajalni-
🚀 Sajalni AI Classifier — Intelligent RAG Email Routing SystemAn automated email classification and routing pipeline designed for the Tunisian Sajalni (CEIR) platform. This system processes incoming user requests, handles dialectal and multilingual complexities, and matches them with the correct administrative response templates using a production-ready Retrieval-Augmented Generation (RAG) architecture.🌟 Key FeaturesMultilingual Semantic Search: Uses intfloat/multilingual-e5-base embeddings to match emails in French, English, and Modern Standard Arabic.Localized Dialectal Anchor (Derja/Arabizi): Built-in semantic boosters to intercept local Tunisian dialect and Arabizi (e.g., nbloki, tserreqli, chkoun) to strictly enforce business compliance rules.Robust Text Chunking: Implements token-aware text chunking via Chonkie to prevent context overflows from signature blocks or long email histories.Hybrid Decision Engine: Uses Qdrant for vector similarity search paired with a local Ollama (Qwen) LLM instance to double-validate categorization decisions.Production-Ready Dockerization: Orchestrated with Docker Compose for seamless replication across development and production environments.🏗️ System ArchitectureExtrait de codegraph TD
    A[Incoming Email] --> B[Chonkie Text Chunking]
    B --> C[E5 Multilingual Embedding]
    C --> D[Qdrant Vector DB + Semantic Boosters]
    D -- Top K Context --> E[Ollama / Qwen LLM]
    E --> F[Final Intent Classification & Email Template]
📁 Repository StructurePlaintextSajjalni/
├── data/
│   └── responses.json         # Administrative intent classes and email templates
├── src/
│   ├── index_kb.py            # Vector database initialization & semantic boosting
│   ├── classifier.py          # RAG classification pipeline (Qdrant + Ollama)
│   └── evaluate.py            # Test suite & accuracy calculator
├── Dockerfile                 # Python environment & embedding caching layer
├── docker-compose.yml         # Qdrant and App multi-container orchestrator
└── requirements.txt           # Python project dependencies
🛠️ Quick StartPrerequisitesDocker & Docker Compose installed and running.Ollama installed locally with the qwen model pulled (ollama pull qwen2.5 or equivalent).1. Spin up the InfrastructureFrom the project root directory, build and launch the containers in detached mode:Bashdocker compose up --build -d
2. Index the Knowledge BaseRun the semantic embedding script inside the application container to populate Qdrant:Bashdocker compose exec sajalni-app python src/index_kb.py
3. Run Evaluation Test SuiteExecute the test benchmark to check classification accuracy:Bashdocker compose exec sajalni-app python src/evaluate.py
📊 Benchmark ResultsThe system achieves 100% classification accuracy across standard validation benchmarks, covering complex cross-lingual technical issues, administrative delays, and native dialect routing:Test #1 (Site Inaccessible): 100% Match (Cosine Similarity $\ge$ 0.88)Test #2 (Derja Dialect Interception): 100% Compliance Route to LANGUE_NON_SUPPORTEETest #3 (Processing Delays): 100% MatchTest #4 (Unsupported Device): 100% Match

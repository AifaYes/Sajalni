# 1. Image Python officielle
FROM python:3.11-slim

# 2. Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models/hf_cache

# 3. Dossier de travail
WORKDIR /app

# 4. Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Installation des dépendances Python
RUN pip install --no-cache-dir qdrant-client sentence-transformers chonkie fastapi uvicorn

# 6. Pré-téléchargement du modèle E5 pour des réponses instantanées
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

# 7. Copie du code et des données
COPY src/ /app/src/
COPY data/ /app/data/

# 8. Commande de lancement
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
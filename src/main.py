from fastapi import FastAPI
from pydantic import BaseModel
import os
from qdrant_client import AsyncQdrantClient  # 🔄 Utilisation du client asynchrone pour utiliser 'await'
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Sajalni AI Classifier API")

# Initialisation des composants
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
# 🔄 On utilise AsyncQdrantClient ici pour correspondre aux appels asynchrones dans les routes
qdrant_client = AsyncQdrantClient(host=qdrant_host, port=6333, check_compatibility=False)
model = SentenceTransformer("intfloat/multilingual-e5-base")

class EmailInput(BaseModel):
    text: str

@app.post("/classify")
async def classify_email(email: EmailInput):
    # 1. Générer l'embedding du texte reçu de n8n
    vector = model.encode(email.text).tolist()
    
    # 2. Rechercher dans Qdrant avec la méthode moderne compatible avec tes vecteurs
    search_result = await qdrant_client.query_points(
        collection_name="sajalni_intents",
        query=vector,  # Ton vecteur d'embedding calculé par sentence-transformers
        limit=1
    )
    
    # (Ici tu intégreras ton double-check Ollama/Qwen par la suite)
    
    # 3. Traiter le résultat renvoyé par query_points
    if search_result and search_result.points:
        best_match = search_result.points[0]
        return {
            "intent": best_match.payload.get("category_id"),          
            "reply_template": best_match.payload.get("reponse_template"), 
            "confidence": best_match.score
        }
    
    return {"intent": "UNKNOWN", "reply_template": "Veuillez patienter...", "confidence": 0.0}
import requests
import re
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from chonkie import TokenChunker
import os
from qdrant_client import QdrantClient

# Si la variable d'environnement existe (dans Docker), on l'utilise, sinon on prend localhost (développement local)
qdrant_host = os.getenv("QDRANT_HOST", "localhost")

client = QdrantClient(host=qdrant_host, port=6333, check_compatibility=False)

chunker = TokenChunker(tokenizer="intfloat/multilingual-e5-base", chunk_size=256)
encoder = SentenceTransformer("intfloat/multilingual-e5-base")

def classify_email(email_content: str):
    chunks = chunker.chunk(email_content)
    main_text = chunks[0].text if chunks else email_content

    # Génération de l'embedding (E5 requiert 'query: ')
    query_vector = encoder.encode(f"query: {main_text}").tolist()

    # Recherche dans Qdrant
    response = client.query_points(
        collection_name="sajalni_intents",
        query=query_vector,
        limit=3
    )
    search_results = response.points

    if not search_results:
        return "UNKNOWN", "No matching category found."

    best_match = search_results[0].payload
    vector_fallback_id = best_match.get("category_id", "UNKNOWN")
    response_template = best_match.get("reponse_template", "Pas de réponse configurée.")
    
    # Construction des options pour le LLM
    context_options = "\n".join([f"- {r.payload.get('category_id')}" for r in search_results])
    
    prompt = f"""You are an expert routing system for the Sajalni mobile registration platform in Tunisia.
Analyze the following user email and select the most appropriate category ID from the list of allowed categories.

User Email:
\"\"\"{main_text}\"\"\"

Allowed Categories (Choose EXACTLY one from this list):
{context_options}

Instructions: Output ONLY the chosen category ID string (e.g., TELEPHONE_VOLE). Do not include any introductory text, explanation, punctuation, or markdown formatting.

Category ID:"""

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
            timeout=10
        )
        llm_decision = res.json().get("response", "").strip()
        
        # Nettoyage strict (enlève les guillemets, points, espaces blancs)
        llm_decision = re.sub(r'[^\w_-]', '', llm_decision)
        
        # Sécurité : Si le LLM a renvoyé du texte vide ou hors sujet, on utilise le choix vectoriel
        if not llm_decision or len(llm_decision) < 3:
            return vector_fallback_id, response_template
            
        return llm_decision, response_template
    except Exception:
        return vector_fallback_id, response_template
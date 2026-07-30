import time
from classifier import classify_email

test_dataset = [
    {
        "email": "Bonjour, j'ai essayé d'accéder au site sajalni pour inscrire mon téléphone mais la page refuse de charger depuis la France. Que faire ?",
        "expected": "SITE_INACCESSIBLE"
    },
    {
        "email": "aslema, brabi telephoni tesra9 chnejem naamel",
        "expected": "LANGUE_NON_SUPPORTEE"
    },
    {
        "email": "Bonjour, j'ai déposé ma demande d'enregistrement avant-hier matin et mon téléphone est toujours bloqué. Combien de temps ça prend ?",
        "expected": "DELAI_TRAITEMENT"
    },
    {
        "email": "Je veux enregistrer ma montre connectée (smartwatch) sur votre plateforme, mais je ne trouve pas l'option.",
        "expected": "EQUIPEMENT_NON_SUPPORTE"
    }
]

print("=== 🧪 LANCEMENT DU TEST DE CLASSIFICATION AUTOMATIQUE ===\n")

correct_predictions = 0

for i, test_case in enumerate(test_dataset):
    start_time = time.time()
    
    predicted_id, response_template = classify_email(test_case["email"])
    
    elapsed = time.time() - start_time
    is_correct = predicted_id == test_case["expected"]
    if is_correct:
        correct_predictions += 1
        
    print(f"📧 Test #{i+1}")
    print(f"Input: {test_case['email'][:80]}...")
    print(f"Attendu: {test_case['expected']} | Prédit: {predicted_id}")
    print(f"Statut: {'✅ SUCCÈS' if is_correct else '❌ ERREUR'}")
    print(f"Temps de réponse: {elapsed:.2f} secondes\n---")

accuracy = (correct_predictions / len(test_dataset)) * 100
print(f"📊 Résultats globaux : Précision de {accuracy:.1f}% ({correct_predictions}/{len(test_dataset)})")
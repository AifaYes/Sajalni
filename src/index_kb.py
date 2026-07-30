import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# 1. Configuration dynamique de l'hôte (Local vs Docker)
# On récupère "QDRANT_HOST" (qui vaut "qdrant" dans docker-compose), sinon "localhost" par défaut
qdrant_host = os.getenv("QDRANT_HOST", "localhost")

# On initialise le client UNIQUE ici au niveau global pour tout le script
client = QdrantClient(host=qdrant_host, port=6333, check_compatibility=False)

def main():
    print("=== 🚀 INITIALISATION DU PIPELINE D'INDEXATION VECTORIELLE ===")
    print(f"🔗 Connexion au serveur Qdrant sur : {qdrant_host}:6333")
    
    # 2. Chargement du modèle d'embedding multilingue de référence (E5)
    print("📥 Chargement du modèle de text-embeddings (intfloat/multilingual-e5-base)...")
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    
    collection_name = "sajalni_intents"

    # 3. Réinitialisation complète de la collection pour purger les anciennes données corrompues
    if client.collection_exists(collection_name=collection_name):
        print(f"🗑️ Ancienne collection '{collection_name}' détectée. Purge en cours...")
        client.delete_collection(collection_name=collection_name)

    print(f"📦 Création de la nouvelle collection vectorielle : '{collection_name}'")
    client.create_collection(
        collection_name=collection_name,
        # Le modèle E5-base génère des vecteurs de dimension 768. La distance Cosine est recommandée.
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # 4. Résolution dynamique du chemin vers la base de connaissances JSON
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "..", "data", "responses.json")

    print(f"📂 Recherche du fichier de configuration : {os.path.abspath(json_path)}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"❌ Erreur critique : Le fichier '{os.path.basename(json_path)}' est introuvable dans le dossier 'data'.\n"
            f"Vérifiez l'arborescence de votre projet Sajjalni."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    # 5. Dictionnaire d'ancrage sémantique (Semantic Boosters)
    # Intègre les exigences métier linguistiques : routage direct de la Derja/Arabizi vers LANGUE_NON_SUPPORTEE
    # Dictionnaire d'ancrage sémantique industriel (Français + Derja + Arabizi + Mots-clés)
   # Dictionnaire d'ancrage sémantique trilingue (Arabe, Français, Anglais) + Jargon
    # La classe LANGUE_NON_SUPPORTEE intercepte exclusivement la Derja, l'Arabizi et les langues tierces
    semantic_boosters = {
        "VERIFICATION_ENREGISTREMENT": (
            "vérifier statut demande preuve capture écran IMEI étoile dièse 06 dièse passeport photo cachet douane "
            "police frontière rejeté refusé invalidé validation correction erreur inscription check status verify registration "
            "proof screenshot passport photo border police stamp rejected invalid correction error sign up التحقق من التسجيل "
            "حالة الطلب إثبات لقطة شاشة جواز السفر ختم شرطة الحدود مرفوض غير صالح تصحيح خطأ رمز"
        ),
        "LANGUE_NON_SUPPORTEE": (
            "Slm nheb nblocki talifoni tserreqli lbera7 fi tounes chkoun nkalem derja tunisien dialecte arabe tunisien "
            "text sms latins arabizi chkoun nkalem a3slema brabi chkoun ya3ref chnoua n3mel y3aishkom louled brabi "
            "yejma3a chwaya 3awnouni language italian german tchello chlaika"
        ),
        "UTILISATION_NON_AUTORISEE_DONNEES": (
            "données personnelles piratage usurpation identité utilisation frauduleuse usurpé vol identité plainte "
            "tribunal ministère intérieur justice procureur personal data hacking identity theft fraudulent use stolen identity "
            "complaint court ministry interior justice prosecutor البيانات الشخصية اختراق سرقة الهوية استخدام احتيالي شكوى "
            "محكمة وزارة الداخلية العدل وكيل الجمهورية"
        ),
        "BLOCAGE_TELEPHONE_VOLE": (
            "téléphone volé perdu bloquer portable enlever puce cambriolage declaration vol cert appareil dérobé agression "
            "stolen phone lost block mobile remove sim card burglary theft declaration stolen device assault حظر الهاتف المسروق "
            "مفقود قفل الجوال إزالة الشريحة سرقة تصريح بالسرقة جهاز مسروق اعتداء اعتراض"
        ),
        "INFO_ENREGISTREMENT_PASSEPORT": (
            "téléphone importé étranger passeport douane bagage voyageur frontalier site sajalni formulaire lien site "
            "imported phone abroad passport customs luggage traveler border website sajalni form link هاتف مستورد الخارج "
            "جواز سفر الجمارك أمتعة مسافر حدودي موقع سجلني استمارة رابط التسجيل"
        ),
        "SITE_INACCESSIBLE": (
            "site sajalni en panne inaccessible ne marche pas de l'étranger site bloqué bug connexion page refuse de charger "
            "tunisie site web en panne connexion error 404 server down la page ne s'ouvre pas website down inaccessible "
            "not working from abroad blocked connection bug page won't load موقع معطل لا يعمل من الخارج محجوب خطأ في الاتصال "
            "الصفحة لا تفتح الخادم متوقف سرفر عاطل"
        ),
        "DELAI_TRAITEMENT": (
            "délai d'attente combien de temps traitement demande encore bloqué pas encore activé maximum jours "
            "ouvrables 3 jours en attente waiting time how long processing request still blocked not activated yet "
            "working days pending فترات الانتظار كم الوقت معالجة الطلب ما زال مغلقا لم يتم التفعيل بعد أيام عمل قيد الانتظار"
        ),
        "CODE_CERT": (
            "code autorisation cert colis postaux douane bureau frontalier douanier récupérer poste envoi importations "
            "authorization code cert postal parcels customs border office custom officer recover mail shipment imports "
            "رمز ترخيص طرود بريدية الجمارك المكتب الحدودي جمركي استرداد البريد شحنة واردات"
        ),
        "ETAPES_ENREGISTREMENT": (
            "étapes suivre procédure guidage tutoriel comment faire enregistrer terminal introduire e-mail lien confirmation "
            "code confirmation steps to follow procedure guide tutorial how to register device enter email confirmation link "
            "خطوات الاتباع إجراءات دليل توجيهي كيفية تسجيل الجهاز إدخال البريد الإلكتروني رابط التأكيد رمز التفعيل"
        ),
        "EQUIPEMENT_NON_SUPPORTE": (
            "montre connectée smartwatch tablette routeur 4g clé wifi pas une option enregistrer montre intelligente "
            "apple watch non supporté puces modem airbox box internet caméras ip connected watch tablet 4g router wifi dongle "
            "not supported internet box ip cameras ساعة ذكية لوحة رقمية تابلت موجه مفتاح واي فاي غير مدعوم مودم كاميرات مراقبة"
        ),
        "DATE_ENTREE_DEPASSEE": (
            "date d'entrée dépassée trois mois 3 mois dépassé délai réglementaire voyage antérieur dépassé les 90 jours "
            "entry date exceeded three months regulatory deadline previous trip exceeded 90 days تاريخ الدخول تجاوز ثلاثة أشهر "
            "المهلة القانونية رحلة سابقة تجاوز تسعين يوما فترة منتهية"
        ),
        "MODALITES_ENREGISTREMENT": (
            "modalités deux manières section dédiée e-bawaba identité numérique citoyen tunisien choix options comment "
            "quelles sont les méthodes registration methods two ways dedicated section digital identity tunisian citizen choices "
            "طرق التسجيل طريقتان القسم المخصص البوابة الإلكترونية الهوية الرقمية مواطن تونسي خيارات"
        ),
        "PASSEPORT_NON_CLAIR": (
            "photo passeport non claire floutée modifiée illisible flou ombre tronquée manque de lumière pas lisible num "
            "caché passport photo not clear blurry modified illegible blur shadow cropped lack of light unreadable hidden number "
            "صورة جواز السفر غير واضحة مموهة معدلة غير مقروءة ضبابية ظل مقطوعة نقص الإضاءة رقم مخفي"
        ),
        "FORMAT_IMAGE_INVALIDE": (
            "format image invalide pdf extension type fichier word docx png jpeg erreur format pas une image "
            "invalid image format file type extension word document error format not an image صيغة صورة غير صالحة امتداد "
            "نوع الملف مستخدم خطأ في التنسيق ليس صورة ملف وورد"
        )
    }

    # 6. Génération des embeddings et préparation des structures de données Qdrant
    points = []
    print("\n--- 📝 Début de l'indexation et de la vectorisation ---")
    
    for idx, item in enumerate(knowledge_base):
        # Lecture flexible pour s'adapter dynamiquement au schéma des attributs du JSON
        c_id = item.get("id") or item.get("category_id")
        c_desc = item.get("description") or item.get("text") or ""
        c_rep = item.get("reponse") or item.get("reponse_template") or item.get("response") or ""
        
        # Injection du booster sémantique correspondant pour forcer le comportement algorithmique attendu
        booster_text = semantic_boosters.get(c_id, "")
        enriched_text = f"{c_id} {c_desc} {booster_text}".strip()
        
        # Le modèle E5 exige STRICTEMENT le préfixe 'passage: ' lors du stockage asymétrique
        text_to_embed = f"passage: {enriched_text}"
        embedding = model.encode(text_to_embed).tolist()
        
        # Construction du payload persistant associé au vecteur
        points.append(
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "category_id": c_id,
                    "reponse_template": c_rep
                }
            )
        )
        print(f"✨ Intent [{c_id}] vectorisé et mappé avec succès.")

    # 7. Chargement (Upsert) des vecteurs dans l'instance Docker Qdrant
    print("\n📤 Injection des paquets de données vectorielles dans la base...")
    client.upsert(collection_name=collection_name, points=points)
    
    print(f"\n✅ BASE VECTORIELLE ET METIER PRÊTE : {len(points)} intentions injectées avec succès dans Qdrant.")

if __name__ == "__main__":
    main()
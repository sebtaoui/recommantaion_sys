# CV Processing Pipeline

Ce projet permet d’analyser automatiquement un lot de CV (en PDF, DOCX ou image) afin d’en extraire des informations structurées, de les enrichir via un LLM (modèle d’intelligence artificielle), et de les indexer pour une recherche vectorielle efficace.

---

##  Fonctionnalité principale

L’API expose un endpoint unique :

### `POST /api/cv/upload-cv-batch`

Ce point d’entrée accepte un **fichier ZIP** contenant plusieurs CV et effectue automatiquement l’ensemble du pipeline suivant :

---

## 🔍 Déroulé complet côté serveur

### 1️ Lecture du ZIP
- Le fichier ZIP est lu **en mémoire** (`zip_bytes`).
- Ouverture avec `zipfile.ZipFile` et itération sur chaque fichier contenu.
- Les **dossiers internes** sont ignorés.
- Chaque **fichier individuel (PDF, DOCX, image)** est traité séparément.

---

### 2️ Extraction du texte brut (`extract_text_from_cv`)
- **PDF** → texte via **PyMuPDF** (avec fallback OCR via **pytesseract** si besoin).  
- **DOCX** → extraction via **python-docx**.  
- **Images (JPG, PNG, etc.)** → OCR avec **pytesseract**.  
- Nettoyage léger du texte : suppression des espaces multiples, normalisation, etc.
- Si le texte extrait est vide → CV **ignoré** (`"Skipping empty CV"`).

---

### 3️ Structuration du contenu (`extract_structured_info`)
- Le texte du CV est envoyé au **LLM** (modèle de langage) via un **prompt spécialisé**.
- Le LLM retourne un **JSON structuré** contenant :
  - Identité (nom, email, téléphone)
  - Éducation
  - Expériences professionnelles
  - Compétences techniques (hard skills)
  - Compétences comportementales (soft skills)
  - Langues, projets, etc.
- Le JSON est **nettoyé et validé** par le modèle **Pydantic `CVInfo`**.
- En cas de JSON invalide ou schéma non conforme → CV **ignoré** et erreur **loggée**.

---

### 4️ Enregistrement dans MongoDB
- Insertion du document structuré dans la collection principale (`collection.insert_one`).
- L’ID Mongo (`inserted_id`) est sauvegardé pour le relier aux étapes suivantes.

---

### 5️ Analyse orientée expérience (`extract_experience_summary`)
- Un **second prompt LLM** est utilisé pour extraire :
  - Le **total d’années d’expérience**
  - Le **niveau de séniorité** (`junior`, `mid`, `senior`)
  - Un **résumé professionnel clair**
  - Les **compétences techniques et comportementales**
- Validation via **Pydantic `CVSummary`**.
- En cas d’échec → CV **ignoré** et erreur **signalée**.

---

### 6️ Préparation du texte pour embedding (`build_text_from_json`)
- Construction d’un **texte pondéré** combinant :
  - Expériences détaillées (répétées 3× pour donner plus de poids)
  - Années d’expérience
  - Niveau de séniorité
  - Hard/Soft skills
  - Éducation et résumé
- Le tout fusionné dans une seule chaîne de texte prête pour l’encodage vectoriel.

---

### 7️ Encodage vectoriel
- Le texte final est encodé avec le modèle **SentenceTransformer**  
  (`all-MiniLM-L6-v2`, `normalize_embeddings=True`).
- Le vecteur obtenu est :
  - Converti en **liste JSON** (pour stockage Mongo)
  - Inséré dans la collection `collection_embedded_data`.

---

### 8️ Indexation FAISS
- Chargement (ou création si inexistant) de l’index :
  - `cv_index.faiss` (vecteurs)
  - `id_map.pkl` (correspondance vecteur ↔ ID Mongo)
- Normalisation L2 du vecteur puis ajout à l’index.
- Mise à jour de la **map d’IDs** et sauvegarde sur disque.

---

### 9 Reporting final
- Comptage du nombre total :
  - CV traités avec succès
  - CV ignorés (erreurs, doublons, fichiers vides)
- Enregistrement des résultats dans une liste `results` :
  - `filename`
  - `status` (`success`, `error`, `skipped`)
  - Messages ou raisons d’échec
- Réponse **JSON** envoyée au client :

```json
{
  "status": "completed",
  "total_processed": 42,
  "total_skipped": 8,
  "message": "Successfully processed 42 CVs, skipped 8",
  "results": [...]
}

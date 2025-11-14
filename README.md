# CV Recommendation System – Architecture & Fonctionnalités

Ce projet offre une chaîne complète pour l’ingestion de CV, leur structuration par LLM et une recommandation hybride combinant recherche vectorielle et re-ranking BERT. L’objectif est de proposer rapidement une short-list de candidats pertinents à partir d’une requête textuelle (job description, profil cible, etc.).

## Architecture globale
- **Backend FastAPI** : expose les endpoints REST, sert l’interface web statique et orchestre le pipeline.
- **Services d’ingestion** : extraction multi-format (PDF, DOCX, images), prompts LLM spécialisés, validation Pydantic.
- **Stockage** : MongoDB pour les documents structurés et les résumés orientés matching.
- **Recherche vectorielle** : SentenceTransformer (SBERT) + index FAISS persistant (`cv_index.faiss` + `id_map.pkl`). Prise en charge des embeddings locaux ou hébergés via Together.ai.
- **Re-ranking** : cross-encoder BERT (`ms-marco-MiniLM-L-6-v2`) avec cache LRU pour accélérer les requêtes récurrentes.
- **Interface web** : formulaire JSON aligné sur l’API, visualisation détaillée des scores et diagnostics.

## Principales fonctionnalités
- **Ingestion batch** (`POST /api/cv/upload-cv-batch`) : import d’un ZIP de CV, structuration, enregistrement Mongo et indexation FAISS (OCR multi-langue + contrôles qualité).
- **Recommandation** (`POST /api/cv/recommend-candidates`) : pipeline en cinq étapes (pré-traitement requête → SBERT/FAISS → re-ranking cross-encoder → fusion des scores → Top 10).
- **Analyse LLM orientée expérience** : résumé analytique, estimation automatique du niveau (junior/mid/senior), extraction hard/soft skills.
- **Interface front** : saisie JSON (poids mots-clés, importance expérience), affichage des composantes de score et export JSON brut.
- **Administration & data cleaning** : stats, purge complète, extraction de numéros de téléphone.

## Flux de traitement – Vue rapide
1. **Upload ZIP** → extraction texte + prompts LLM → stockage Mongo + embeddings FAISS.
2. **Requête utilisateur** → normalisation & analyse → FAISS Top K (configurable) → re-ranking cross-encoder.
3. **Fusion des scores** (embedding, cross-encoder, mots-clés, expérience) → Top 10 retourné avec diagnostics complets.

---

## Configuration essentielle

| Variable | Rôle | Valeur par défaut |
|----------|------|-------------------|
| `MONGO_URI`, `DB_NAME` | Connexion MongoDB | `mongodb://localhost:27017/`, `cv_recommendation_db` |
| `TOGETHER_API_KEY`, `TOGETHER_MODEL` | Accès LLM pour la structuration | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| `EMBEDDING_MODEL` | Modèle SentenceTransformer ou Together.ai | `sentence-transformers/all-mpnet-base-v2` |
| `RERANKER_MODEL` | Cross-encoder de re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `FAISS_PRESELECTION_K` | Taille pré-sélection FAISS | `100` |
| `FUSION_*` | Poids des composantes (embedding / cross-encoder / keywords / expérience) | `0.45 / 0.45 / 0.05 / 0.05` |
| `EMBEDDING_PROVIDER` | `local` ou `together` | `local` |
| `OCR_LANGUAGES` | Langues OCR Tesseract (codes concaténés) | `eng+fra` |
| `OCR_MIN_CONFIDENCE` | Seuil de confiance OCR (0-100) | `35.0` |
| `CALIBRATION_PROFILE_PATH` | Profil de calibration JSON | `calibration/weights.json` |

Les poids sont normalisés automatiquement et peuvent être ajustés à l’aide d’un jeu d’or (précision@K, nDCG). L’événement `startup_event` chauffe les modèles pour limiter la latence du premier appel, et un cache LRU évite de recalculer les scores cross-encoder répétés.

### Calibration guidée
Un script dédié permet de calibrer automatiquement les poids sur un corpus annoté :

```bash
python scripts/calibrate_weights.py --dataset data/annotated_samples.jsonl --grid-step 0.05
```

Le profil généré est enregistré dans `calibration/weights.json` puis chargé automatiquement au démarrage.

---

## Détails du pipeline d’ingestion

### Pipeline complet

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
- **PDF** → texte via **PyMuPDF** (avec fallback OCR multi-langue via **pytesseract** configurable).  
- **DOCX** → extraction via **python-docx**.  
- **Images (JPG, PNG, etc.)** → OCR avec contrôle de confiance (seuil `OCR_MIN_CONFIDENCE`).  
- Ré-exécution OCR avec paramètres optimisés si la confiance est basse.  
- Nettoyage du texte (normalisation unicode, ponctuation). CV vides ou non reconnaissables → ignorés.

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
- Post-traitements automatiques : déduplication des expériences, vérification des dates et ajout éventuel de `validationWarnings`.
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
- Validation via **Pydantic `CVSummary`** + contrôles (valeur d’années non négative, duplication de phrases).
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
- Le texte final est encodé avec un modèle **SentenceTransformer** avancé (par défaut `all-mpnet-base-v2`) ou un modèle hébergé via Together.ai.
- Les vecteurs sont normalisés puis :
  - Convertis en **liste JSON** (pour stockage Mongo)
  - Insérés dans la collection `collection_embedded_data`.

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
- Réponse **JSON** envoyée au client (incluant les `validationWarnings` éventuels pour suivi de qualité) :

```json
{
  "status": "completed",
  "total_processed": 42,
  "total_skipped": 8,
  "message": "Successfully processed 42 CVs, skipped 8",
  "results": [...]
}

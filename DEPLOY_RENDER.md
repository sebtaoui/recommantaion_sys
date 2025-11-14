# Déploiement sur Render

Ce guide explique comment déployer le système de recommandation de CV sur Render.

## Prérequis

- Un compte Render (gratuit disponible)
- Un dépôt GitHub avec le code du projet

## Étapes de déploiement

### 1. Préparer le dépôt GitHub

Assure-toi que ton code est poussé sur GitHub avec tous les fichiers nécessaires :
- `Dockerfile`
- `requirements.txt`
- `render.yaml` (optionnel mais recommandé)
- Tous les fichiers de l'application

### 2. Créer un service MongoDB sur Render

1. Connecte-toi à [Render Dashboard](https://dashboard.render.com)
2. Clique sur **"New +"** → **"PostgreSQL"** ou **"MongoDB"** (si disponible)
3. Si MongoDB n'est pas disponible directement, utilise **"MongoDB Atlas"** (gratuit) ou un service externe
4. Note l'URI de connexion MongoDB (ex: `mongodb+srv://user:pass@cluster.mongodb.net/`)

### 3. Créer le service Web (API)

#### Option A : Utiliser render.yaml (Recommandé)

1. Dans Render Dashboard, clique sur **"New +"** → **"Blueprint"**
2. Connecte ton dépôt GitHub
3. Render détectera automatiquement `render.yaml` et créera les services configurés
4. Configure les variables d'environnement dans le dashboard si nécessaire

#### Option B : Configuration manuelle

1. Clique sur **"New +"** → **"Web Service"**
2. Connecte ton dépôt GitHub
3. Configure le service :
   - **Name**: `cv-recommendation-api`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Docker Context**: `.`
   - **Plan**: `Starter` (gratuit avec limitations) ou `Standard` (payant)
   - **Health Check Path**: `/`

4. **Variables d'environnement** (à configurer dans le dashboard) :
   ```
   MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/
   DB_NAME=cv_recommendation_db
   TOGETHER_API_KEY=ton_api_key (si tu utilises Together.ai)
   TOGETHER_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
   EMBEDDING_PROVIDER=local
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   ```

5. Clique sur **"Create Web Service"**

### 4. Configuration MongoDB

Si tu utilises MongoDB Atlas (recommandé pour Render) :

1. Crée un compte sur [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Crée un cluster gratuit (M0)
3. Configure l'accès réseau (ajoute `0.0.0.0/0` pour autoriser toutes les IPs, ou l'IP de Render)
4. Crée un utilisateur avec mot de passe
5. Copie l'URI de connexion et utilise-la comme `MONGO_URI` dans Render

### 5. Vérifier le déploiement

1. Une fois déployé, Render fournira une URL HTTPS automatique (ex: `https://cv-recommendation-api.onrender.com`)
2. Teste l'API :
   - `https://ton-url.onrender.com/` → Devrait afficher l'interface web
   - `https://ton-url.onrender.com/api/cv/stats` → Devrait retourner les statistiques

### 6. Limitations du plan gratuit Render

- **Sleep après inactivité** : Le service se met en veille après 15 minutes d'inactivité
- **Limite de RAM** : 512 MB
- **Limite de CPU** : 0.1 CPU partagé
- **Temps de démarrage** : Peut prendre 30-60 secondes après le réveil

### 7. Améliorer les performances

Pour éviter le "cold start" :
- Utilise un service externe pour ping ton API toutes les 5 minutes (ex: UptimeRobot gratuit)
- Ou upgrade vers un plan payant qui ne dort jamais

## Notes importantes

- **HTTPS** : Render fournit automatiquement HTTPS avec certificat SSL valide
- **Variables sensibles** : Ne commite JAMAIS tes clés API dans le code. Utilise les variables d'environnement Render
- **Persistance des données** : Les fichiers FAISS (`cv_index.faiss`, `id_map.pkl`) sont stockés dans le système de fichiers éphémère. Pour la persistance, considère utiliser un service de stockage externe (S3, etc.)
- **Logs** : Accède aux logs via le dashboard Render ou l'API Render

## Dépannage

### Le service ne démarre pas
- Vérifie les logs dans Render Dashboard
- Assure-toi que `MONGO_URI` est correctement configuré
- Vérifie que le port 8000 est bien exposé dans le Dockerfile

### Erreurs de connexion MongoDB
- Vérifie que l'IP de Render est autorisée dans MongoDB Atlas
- Vérifie les credentials MongoDB
- Assure-toi que l'URI MongoDB est correcte (format `mongodb+srv://` pour Atlas)

### Timeout ou erreurs 502
- Le service peut être en train de démarrer (cold start)
- Vérifie les logs pour des erreurs de mémoire
- Considère upgrade vers un plan avec plus de ressources

## Support

Pour plus d'aide :
- [Documentation Render](https://render.com/docs)
- [Documentation MongoDB Atlas](https://docs.atlas.mongodb.com/)


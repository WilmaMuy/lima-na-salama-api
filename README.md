# Lima na Salama - Backend API

Backend FastAPI pour le diagnostic des maladies des plantes par intelligence artificielle.

## 🚀 Installation et démarrage

### Prérequis
- Python 3.11+
- Modèle TensorFlow `Diagnostic_Plantes.h5`

### Installation locale

1. **Cloner et naviguer vers le dossier backend**
```bash
cd backend
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Placer le modèle**
Copiez votre fichier `Diagnostic_Plantes.h5` dans le dossier `backend/`

5. **Démarrer le serveur**
```bash
python main.py
```

L'API sera disponible sur `http://localhost:8000`

### Installation avec Docker

1. **Construire l'image**
```bash
docker-compose up --build
```

2. **Démarrer en arrière-plan**
```bash
docker-compose up -d
```

## 📚 Documentation API

### Endpoints principaux

#### `GET /` - Informations générales
```json
{
  "message": "Lima na Salama - API de Diagnostic des Plantes",
  "version": "1.0.0",
  "status": "active",
  "model_loaded": true
}
```

#### `GET /health` - État de santé
```json
{
  "status": "healthy",
  "timestamp": "2025-01-15T10:30:00",
  "model_status": "loaded",
  "classes_count": 8
}
```

#### `POST /diagnose` - Diagnostic d'image
**Paramètres:** 
- `file`: Image de la plante (multipart/form-data)

**Réponse:**
```json
{
  "success": true,
  "disease": "Cercosporiose grise du maïs",
  "confidence": 95.67,
  "description": "Maladie fongique causée par...",
  "timestamp": "2025-01-15T10:30:00",
  "all_predictions": [
    {
      "disease": "Cercosporiose grise du maïs",
      "confidence": 95.67
    },
    {
      "disease": "Mildiou de la tomate",
      "confidence": 3.21
    }
  ]
}
```

#### `GET /diseases` - Liste des maladies
```json
{
  "diseases": [
    {
      "name": "Cercosporiose grise du maïs",
      "description": "Maladie fongique causée par..."
    }
  ],
  "total_count": 10
}
```

## 🧪 Tests

### Test automatique
```bash
python test_api.py
```

### Test manuel avec curl
```bash
# Test de santé
curl http://localhost:8000/health

# Test de diagnostic
curl -X POST "http://localhost:8000/diagnose" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

## 🔧 Configuration

### Variables d'environnement
- `MODEL_PATH`: Chemin vers le modèle (défaut: `Diagnostic_Plantes.h5`)
- `HOST`: Adresse d'écoute (défaut: `0.0.0.0`)
- `PORT`: Port d'écoute (défaut: `8000`)

### Adaptation du modèle
Modifiez les constantes dans `main.py`:
- `DISEASE_CLASSES`: Liste des classes de votre modèle
- `DISEASE_DESCRIPTIONS`: Descriptions des maladies
- `target_size` dans `preprocess_image()`: Taille d'entrée de votre modèle

## 📱 Intégration avec l'app mobile

L'API est automatiquement intégrée dans l'application React Native via le service `apiService.ts`.

### Configuration de l'URL
Dans `services/apiService.ts`, modifiez:
```typescript
const DIAGNOSTIC_API_URL = 'http://votre-serveur:8000';
```

## 🐛 Dépannage

### Erreur "Modèle non trouvé"
- Vérifiez que `Diagnostic_Plantes.h5` est dans le dossier backend
- Vérifiez les permissions de lecture du fichier

### Erreur de mémoire
- Réduisez la taille des images d'entrée
- Augmentez la mémoire disponible pour le conteneur Docker

### Erreur CORS
- Vérifiez la configuration CORS dans `main.py`
- En production, spécifiez les domaines autorisés

## 📈 Monitoring

### Logs
Les logs sont disponibles dans la console ou dans `/app/logs` (Docker)

### Métriques
- Temps de réponse des prédictions
- Taux de succès des diagnostics
- Utilisation mémoire du modèle

## 🚀 Déploiement en production

### Recommandations
1. Utiliser un serveur ASGI comme Gunicorn + Uvicorn
2. Configurer un reverse proxy (Nginx)
3. Activer HTTPS
4. Limiter les domaines CORS
5. Ajouter l'authentification si nécessaire
6. Monitorer les performances

### Exemple de déploiement
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
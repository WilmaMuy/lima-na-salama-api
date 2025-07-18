from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageEnhance, ImageFilter
import io
import logging
from typing import Dict, List
import uvicorn
from datetime import datetime
import os
import cv2

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lima na Salama - API de Diagnostic",
    description="API pour le diagnostic des maladies des plantes par IA",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes depuis l'app mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales pour le modèle
model = None
class_names = []

# Classes de maladies 
DISEASE_CLASSES = [
    "Brûlure foliaire du Maïs",
    "Chenille légionnaire du Maïs",
    "Curvulariose du Maïs",
    "Rouille commune du Maïs",
    "Feuille Saine du Maïs",
    "Striure du Maïs",
    "Tache grise foliaire du Maïs",
    
]

# Descriptions des maladies
DISEASE_DESCRIPTIONS = {
  "Brûlure foliaire du Maïs": "Maladie fongique provoquant des lésions nécrotiques allongées sur les feuilles, débutant souvent par des taches d’eau brunâtres qui s’étendent rapidement.",
  "Chenille légionnaire du Maïs": "Ravageur destructeur (Spodoptera frugiperda) creusant des trous irréguliers dans les feuilles, les tiges et les épis, réduisant considérablement le rendement.",
  "Curvulariose du Maïs": "Maladie fongique causée par Curvularia spp., caractérisée par des taches brunes à noires entourées de halos jaunes sur les feuilles.",
  "Rouille commune du Maïs": "Maladie fongique causée par Puccinia sorghi, produisant des pustules orange-brun en relief sur les deux faces des feuilles.",
  "Feuille Saine du Maïs": "Feuille de maïs exempte de symptômes visibles de maladies ou de ravageurs, présentant une couleur verte uniforme et une texture intacte.",
  "Striure du Maïs": "Symptôme viral ou carentiel apparaissant sous forme de lignes ou stries jaunes à blanchâtres sur les feuilles, affectant la photosynthèse.",
  "Tache grise foliaire du Maïs": "Maladie fongique causée par Cercospora zeae-maydis, caractérisée par des lésions grisâtres allongées entre les nervures des feuilles."
}


def load_model():
    """Charge le modèle de diagnostic des plantes"""
    global model, class_names
    try:
        model_path = r"C:\Users\User\Desktop\Application\Lima\backend\Diagnostic_Plantes.h5"
        if not os.path.exists(model_path):
            logger.error(f"Modèle non trouvé: {model_path}")
            return False
        
        model = tf.keras.models.load_model(model_path)
        class_names = DISEASE_CLASSES
        logger.info(f"Modèle chargé avec succès. Classes: {len(class_names)}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Préprocesse l'image avec amélioration de qualité pour le modèle"""
    try:
        # Étape 1: Validation et conversion de base
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Étape 2: Amélioration de la qualité
        image = enhance_image_quality(image)
        
        # Étape 3: Redimensionnement intelligent
        image = smart_resize(image)
        
        # Étape 4: Normalisation et augmentation
        image_array = apply_preprocessing_pipeline(image)
        
        return image_array
    except Exception as e:
        logger.error(f"Erreur preprocessing: {str(e)}")
        raise HTTPException(status_code=400, detail="Erreur lors du traitement de l'image")

def enhance_image_quality(image: Image.Image) -> Image.Image:
    """Améliore la qualité de l'image avant le diagnostic"""
    try:
        # Convertir en array numpy pour OpenCV
        img_array = np.array(image)
        
        # 1. Réduction du bruit
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # 2. Amélioration du contraste adaptatif
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Reconvertir en PIL Image
        enhanced_image = Image.fromarray(img_array)
        
        # 3. Amélioration de la netteté
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced_image)
        enhanced_image = sharpness_enhancer.enhance(1.2)
        
        # 4. Ajustement de la saturation pour les plantes
        color_enhancer = ImageEnhance.Color(enhanced_image)
        enhanced_image = color_enhancer.enhance(1.1)
        
        return enhanced_image
    except Exception as e:
        logger.warning(f"Erreur amélioration qualité: {str(e)}")
        return image

def smart_resize(image: Image.Image, target_size: tuple = (224, 224)) -> Image.Image:
    """Redimensionnement intelligent avec préservation du ratio d'aspect"""
    try:
        # Calculer le ratio d'aspect
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Calculer le ratio de redimensionnement
        ratio = min(target_width / original_width, target_height / original_height)
        
        # Nouvelles dimensions
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Redimensionner avec un filtre de haute qualité
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Créer une image avec padding si nécessaire
        final_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # Centrer l'image redimensionnée
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        final_image.paste(resized_image, (paste_x, paste_y))
        
        return final_image
    except Exception as e:
        logger.warning(f"Erreur redimensionnement: {str(e)}")
        return image.resize(target_size, Image.Resampling.LANCZOS)

def apply_preprocessing_pipeline(image: Image.Image) -> np.ndarray:
    """Pipeline de prétraitement final pour le modèle"""
    try:
        # Redimensionner l'image (adapter selon votre modèle)
        target_size = (224, 224)
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convertir en array numpy
        image_array = np.array(image, dtype=np.float32)
        
        # Normalisation avancée
        # Option 1: Normalisation standard [0, 1]
        image_array = image_array / 255.0
        
        # Option 2: Normalisation ImageNet (décommentez si votre modèle l'utilise)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array - mean) / std
        
        # Augmentation de données légère pour améliorer la robustesse
        image_array = apply_light_augmentation(image_array)
        
        # Ajouter la dimension batch
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        logger.error(f"Erreur pipeline preprocessing: {str(e)}")
        raise

def apply_light_augmentation(image_array: np.ndarray) -> np.ndarray:
    """Applique une légère augmentation pour améliorer la robustesse"""
    try:
        # Légère variation de luminosité (±5%)
        brightness_factor = np.random.uniform(0.95, 1.05)
        image_array = np.clip(image_array * brightness_factor, 0.0, 1.0)
        
        # Légère variation de contraste (±3%)
        contrast_factor = np.random.uniform(0.97, 1.03)
        mean = np.mean(image_array)
        image_array = np.clip((image_array - mean) * contrast_factor + mean, 0.0, 1.0)
        
        return image_array
    except Exception as e:
        logger.warning(f"Erreur augmentation: {str(e)}")
        return image_array

def validate_image_for_diagnosis(image: Image.Image) -> Dict[str, any]:
    """Valide la qualité de l'image pour le diagnostic"""
    validation_result = {
        "is_valid": True,
        "quality_score": 100,
        "issues": [],
        "recommendations": []
    }
    
    try:
        # Vérifier les dimensions
        width, height = image.size
        if width < 224 or height < 224:
            validation_result["is_valid"] = False
            validation_result["quality_score"] -= 30
            validation_result["issues"].append("Résolution trop faible")
            validation_result["recommendations"].append("Utilisez une image d'au moins 224x224 pixels")
        
        # Vérifier le format
        if image.mode not in ['RGB', 'RGBA']:
            validation_result["quality_score"] -= 20
            validation_result["issues"].append("Format de couleur non optimal")
        
        # Analyser la luminosité
        img_array = np.array(image.convert('RGB'))
        brightness = np.mean(img_array)
        
        if brightness < 50:
            validation_result["quality_score"] -= 25
            validation_result["issues"].append("Image trop sombre")
            validation_result["recommendations"].append("Améliorez l'éclairage")
        elif brightness > 200:
            validation_result["quality_score"] -= 20
            validation_result["issues"].append("Image surexposée")
            validation_result["recommendations"].append("Réduisez l'exposition")
        
        # Analyser le contenu vert (végétation)
        green_channel = img_array[:, :, 1]
        red_channel = img_array[:, :, 0]
        blue_channel = img_array[:, :, 2]
        
        green_dominance = np.mean(green_channel > red_channel) + np.mean(green_channel > blue_channel)
        
        if green_dominance < 0.3:
            validation_result["quality_score"] -= 15
            validation_result["issues"].append("Peu de végétation visible")
            validation_result["recommendations"].append("Cadrez sur la plante malade")
        
        # Score final
        if validation_result["quality_score"] < 60:
            validation_result["is_valid"] = False
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Erreur validation image: {str(e)}")
        return {
            "is_valid": True,  # Par défaut, accepter l'image
            "quality_score": 80,
            "issues": ["Validation automatique non disponible"],
            "recommendations": []
        }

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage"""
    success = load_model()
    if not success:
        logger.warning("Impossible de charger le modèle au démarrage")

@app.get("/")
async def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Lima na Salama - API de Diagnostic des Plantes",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model is not None else "not_loaded",
        "classes_count": len(class_names)
    }

@app.post("/diagnose")
async def diagnose_plant(file: UploadFile = File(...)):
    """
    Diagnostique une maladie de plante à partir d'une image avec prétraitement avancé
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Lire l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Valider la qualité de l'image
        validation = validate_image_for_diagnosis(image)
        logger.info(f"Validation image - Score: {validation['quality_score']}%, Valide: {validation['is_valid']}")
        
        # Préprocesser l'image
        processed_image = preprocess_image(image)
        
        # Faire la prédiction
        predictions = model.predict(processed_image)
        
        # Obtenir la classe prédite et la confiance
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index]) * 100
        
        # Obtenir le nom de la maladie
        disease_name = class_names[predicted_class_index]
        description = DISEASE_DESCRIPTIONS.get(disease_name, "Description non disponible")
        
        # Préparer la réponse
        result = {
            "success": True,
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "description": description,
            "image_quality": validation,
            "timestamp": datetime.now().isoformat(),
            "all_predictions": [
                {
                    "disease": class_names[i],
                    "confidence": round(float(predictions[0][i]) * 100, 2)
                }
                for i in range(len(class_names))
            ]
        }
        
        logger.info(f"Diagnostic réalisé: {disease_name} ({confidence:.2f}%)")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors du diagnostic: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors du diagnostic: {str(e)}")

@app.post("/validate-image")
async def validate_image_quality(file: UploadFile = File(...)):
    """
    Valide la qualité d'une image avant diagnostic
    """
    # Vérifier le type de fichier
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        # Lire l'image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Valider la qualité
        validation_result = validate_image_for_diagnosis(image)
        
        return {
            "validation": validation_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur validation image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la validation: {str(e)}")
@app.get("/diseases")
async def get_diseases():
    """Retourne la liste des maladies que le modèle peut diagnostiquer"""
    diseases_info = []
    for disease in class_names:
        diseases_info.append({
            "name": disease,
            "description": DISEASE_DESCRIPTIONS.get(disease, "Description non disponible")
        })
    
    return {
        "diseases": diseases_info,
        "total_count": len(diseases_info)
    }

@app.get("/disease/{disease_name}")
async def get_disease_info(disease_name: str):
    """Retourne les informations détaillées sur une maladie"""
    if disease_name not in class_names:
        raise HTTPException(status_code=404, detail="Maladie non trouvée")
    
    return {
        "name": disease_name,
        "description": DISEASE_DESCRIPTIONS.get(disease_name, "Description non disponible"),
        "index": class_names.index(disease_name)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
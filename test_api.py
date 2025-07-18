import requests
import json
from pathlib import Path

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_health():
    """Test de l'endpoint de santé"""
    response = requests.get(f"{BASE_URL}/health")
    print("=== Test Health ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_diseases():
    """Test de l'endpoint des maladies"""
    response = requests.get(f"{BASE_URL}/diseases")
    print("=== Test Diseases ===")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Nombre de maladies: {data['total_count']}")
    for disease in data['diseases'][:3]:  # Afficher les 3 premières
        print(f"- {disease['name']}")
    print()

def test_diagnosis(image_path):
    """Test du diagnostic avec une image"""
    if not Path(image_path).exists():
        print(f"Image non trouvée: {image_path}")
        return
    
    print("=== Test Diagnosis ===")
    with open(image_path, 'rb') as f:
        files = {'file': ('test_image.jpg', f, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/diagnose", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Maladie détectée: {data['disease']}")
        print(f"Confiance: {data['confidence']}%")
        print(f"Description: {data['description'][:100]}...")
    else:
        print(f"Erreur: {response.text}")
    print()

if __name__ == "__main__":
    print("🧪 Test de l'API Lima na Salama\n")
    
    # Tests
    test_health()
    test_diseases()
    
    # Test avec une image (remplacer par le chemin de votre image de test)
    test_image_path = r"C:\Users\User\Desktop\Mémoire\Lima_na _Salama\assets\images\image_processing20220902-2740616-dqes3.jpg"
    test_diagnosis(test_image_path)
    
    print("✅ Tests terminés")
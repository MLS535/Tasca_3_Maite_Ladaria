import requests
import json

BASE_URL = "http://127.0.0.1:5000/predict"

MODELS = [
    "logistic_regression",
    "svm",
    "knn",
    "decision_tree"
]

# Dos ejemplos de pingüinos (peticiones)
samples = [
    {
        "culmen_length_mm": 39.1,
        "culmen_depth_mm": 18.7,
        "flipper_length_mm": 181,
        "body_mass_g": 3750,
        "island": "Torgersen",
        "sex": "Male"
    },
    {
        "culmen_length_mm": 50.0,
        "culmen_depth_mm": 15.3,
        "flipper_length_mm": 230,
        "body_mass_g": 5000,
        "island": "Biscoe",
        "sex": "Male"
    }
]


def send_request(model_name, data):
    url = f"{BASE_URL}/{model_name}"
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": response.status_code,
            "message": response.text
        }


if __name__ == "__main__":
    print("=== CLIENTE DE PREDICCIÓN DE PINGÜINOS ===\n")

    for model in MODELS:
        print(f"\n--- Modelo: {model} ---")
        for i, sample in enumerate(samples, start=1):
            print(f"\nPetición {i}:")
            print(sample)

            result = send_request(model, sample)
            print("Respuesta:")
            print(result)

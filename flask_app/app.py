from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Cargar modelos serializados
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

models = {
    "logistic_regression": joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib")),
    "svm": joblib.load(os.path.join(MODELS_DIR, "svm.joblib")),
    "knn": joblib.load(os.path.join(MODELS_DIR, "knn.joblib")),
    "decision_tree": joblib.load(os.path.join(MODELS_DIR, "decision_tree.joblib")),
}


# -----------------------------
# Función común de predicción
# -----------------------------
def predict(bundle, input_data):
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    dict_vectorizer = bundle["dict_vectorizer"]
    scaler = bundle["scaler"]
    num_features = bundle["num_features"]

    # Variables numéricas
    X_num = np.array([[input_data[f] for f in num_features]])

    # Variables categóricas
    X_cat = dict_vectorizer.transform([{
        "island": input_data["island"],
        "sex": input_data["sex"]
    }])

    # Escalado SOLO de numéricas
    X_num_scaled = scaler.transform(X_num)

    # Concatenar
    X_final = np.hstack((X_num_scaled, X_cat))

    # Predicción
    pred = model.predict(X_final)[0]
    species = label_encoder.inverse_transform([pred])[0]

    return species


# -----------------------------
# Endpoints
# -----------------------------
@app.route("/predict/<model_name>", methods=["POST"])
def predict_model(model_name):
    if model_name not in models:
        return jsonify({"error": "Modelo no disponible"}), 404

    input_data = request.get_json()

    try:
        species = predict(models[model_name], input_data)
        return jsonify({
            "model": model_name,
            "prediction": species
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    return "API de predicción de pingüinos (Palmer Penguins)"


if __name__ == "__main__":
    app.run(debug=True)

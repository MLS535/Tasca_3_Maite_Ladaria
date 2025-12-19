import sys
import os
import joblib

# Añadimos notebook/ al PYTHONPATH para reutilizar data_utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import load_and_prepare_data

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_knn(data, k=5):
    """
    Entrena un modelo KNN (k-Nearest Neighbours)
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = KNeighborsClassifier(
        n_neighbors=k,
        metric="minkowski",
        p=2  # Distancia euclídea
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Evaluación
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"=== KNN (k={k}) ===")
    print("Accuracy train:", accuracy_score(y_train, y_train_pred))
    print("Accuracy test:", accuracy_score(y_test, y_test_pred))

    print("\nConfusion matrix (test):")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nClassification report (test):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=data["label_encoder"].classes_
        )
    )

    return model


def save_model(model, data, model_name):
    """
    Guarda el modelo KNN y todo el pipeline necesario con joblib
    """
    models_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "models")
    )
    os.makedirs(models_path, exist_ok=True)

    bundle = {
        "model": model,
        "label_encoder": data["label_encoder"],
        "dict_vectorizer": data["dict_vectorizer"],
        "scaler": data["scaler"],
        "num_features": data["num_features"]
    }

    model_file = os.path.join(models_path, f"{model_name}.joblib")
    joblib.dump(bundle, model_file)

    print(f"\nModelo guardado correctamente en:\n{model_file}")


if __name__ == "__main__":
    # 1️⃣ Cargar y preparar datos
    data = load_and_prepare_data()

    # 2️⃣ Entrenar modelo KNN
    model = train_knn(data, k=5)

    # 3️⃣ Serializar modelo
    save_model(model, data, "knn")

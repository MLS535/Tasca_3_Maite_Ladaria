import sys
import os
import joblib

# Añadimos notebook/ al PYTHONPATH para reutilizar data_utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import load_and_prepare_data

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_svm(data):
    """
    Entrena un modelo SVM (Support Vector Machine) con kernel lineal
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = SVC(
        kernel="linear",
        C=1.0,
        random_state=1
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Evaluación
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("=== Support Vector Machine (SVM) ===")
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
    Guarda el modelo SVM y todo el pipeline necesario con joblib
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

    # 2️⃣ Entrenar modelo SVM
    model = train_svm(data)

    # 3️⃣ Serializar modelo
    save_model(model, data, "svm")

import sys
import os
import joblib

# Añadimos notebook/ al PYTHONPATH para reutilizar data_utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import load_and_prepare_data

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_decision_tree(data, max_depth=4, criterion="gini"):
    """
    Entrena un modelo de Árbol de Decisión
    """
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    model = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=1
    )

    # Entrenamiento
    model.fit(X_train, y_train)

    # Evaluación
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"=== Decision Tree (criterion={criterion}, max_depth={max_depth}) ===")
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
    Guarda el Árbol de Decisión y todo el pipeline necesario con joblib
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

    # 2️⃣ Entrenar Árbol de Decisión
    model = train_decision_tree(data, max_depth=4, criterion="gini")

    # 3️⃣ Serializar modelo
    save_model(model, data, "decision_tree")

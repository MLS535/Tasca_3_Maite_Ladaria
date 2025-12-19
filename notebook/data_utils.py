import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer


def load_and_prepare_data(test_size=0.2, random_state=1):
    # 📍 Ruta del CSV definida aquí
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base_dir, "dataset", "penguins_size.csv")

    # Cargar dataset
    df = pd.read_csv(csv_path)

    # Eliminar NA
    df = df.dropna()

    # Variable objetivo
    le = LabelEncoder()
    y = le.fit_transform(df["species"])

    # Variables
    num_features = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g"
    ]

    cat_features = [
        "island",
        "sex"
    ]

    X_num = df[num_features]
    X_cat = df[cat_features]

    # One-hot encoding
    dv = DictVectorizer(sparse=False)
    X_cat_oh = dv.fit_transform(X_cat.to_dict(orient="records"))

    X = np.hstack((X_num.values, X_cat_oh))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    n_num = len(num_features)

    scaler.fit(X_train[:, :n_num])
    X_train[:, :n_num] = scaler.transform(X_train[:, :n_num])
    X_test[:, :n_num] = scaler.transform(X_test[:, :n_num])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": le,
        "num_features": num_features,
        "dict_vectorizer": dv,
        "scaler": scaler,
    }

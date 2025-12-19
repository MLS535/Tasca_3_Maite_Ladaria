import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1️⃣ Carregar dataset
df = pd.read_csv("../dataset/penguins_size.csv")

# 2️⃣ Eliminar files amb NA
df = df.dropna()

# 3️⃣ Variable objectiu (species) → numèrica
le = LabelEncoder()
y = le.fit_transform(df["species"])

# 4️⃣ Separar variables numèriques i categòriques
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

# 5️⃣ One-hot encoding de les variables categòriques amb DictVectorizer
dv = DictVectorizer(sparse=False)
X_cat_dict = X_cat.to_dict(orient="records")
X_cat_oh = dv.fit_transform(X_cat_dict)

# 6️⃣ Concatenar variables numèriques + categòriques
X = np.hstack((X_num.values, X_cat_oh))

# 7️⃣ Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

# 8️⃣ Normalització (només variables numèriques)
scaler = StandardScaler()

X_train_num = X_train[:, :len(num_features)]
X_test_num = X_test[:, :len(num_features)]

scaler.fit(X_train_num)

X_train[:, :len(num_features)] = scaler.transform(X_train_num)
X_test[:, :len(num_features)] = scaler.transform(X_test_num)

# 9️⃣ Entrenament del model
lr = OneVsRestClassifier(
    LogisticRegression(
        C=100.0,
        random_state=1,
        solver="lbfgs",
        max_iter=200
    )
)

lr.fit(X_train, y_train)

# 🔟 Accuracy bàsica
print("Accuracy train:", lr.score(X_train, y_train))
print("Accuracy test:", lr.score(X_test, y_test))

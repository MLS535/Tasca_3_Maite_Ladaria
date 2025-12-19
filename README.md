# Tasca 3 – Palmer Penguins (Maite Ladaria)

En aquesta tasca del mòdul **Sistemes d’Aprenentatge Automàtic**, he implementat i desplegat diferents **models de classificació supervisada** utilitzant el dataset *Palmer Penguins*, seguint el mateix enfocament que al cas pràctic del dataset Iris.

El projecte cobreix tot el flux habitual d’un problema de classificació, des del preprocessament de les dades fins al desplegament d’un servei web per fer prediccions.

El projecte inclou:
- Preprocessament complet de les dades
- Entrenament de **4 models de classificació**
- Serialització dels models
- Desplegament d’un **servei web amb Flask**
- Un client Python que consumeix el servei
- Gràfiques a la carpeta `Images`

## Dataset

S’utilitza el dataset **Palmer Penguins**, descarregat des de Kaggle (`penguins_size.csv`).

Variables del dataset:
- `species` → **variable objectiu** (Adelie, Chinstrap, Gentoo)
- `island` → Dream, Torgersen, Biscoe
- `culmen_length_mm`
- `culmen_depth_mm`
- `flipper_length_mm`
- `body_mass_g`
- `sex` → Male / Female

Per simplificar el treball, s’eliminen les files que contenen valors `NA`.

## Preprocessament de les dades

El preprocessament és comú per a tots els models i es realitza al mòdul `notebook/data_utils.py`.

Inclou els següents passos:
- Eliminació de valors nuls
- Divisió del dataset:
  - **80% per a entrenament**
  - **20% per a prova**
- Codificació de la variable objectiu (`species`) amb `LabelEncoder`
- Codificació *one-hot* de les variables categòriques (`island`, `sex`) amb `DictVectorizer`
- Normalització de les variables numèriques amb `StandardScaler`
  - mitjana = 0
  - desviació típica = 1
  - l’escalat s’ajusta **només amb el conjunt d’entrenament** i s’aplica a entrenament i prova

Aquest procés evita *data leakage* i garanteix la coherència entre entrenament i predicció.

## Models implementats

S’han entrenat i avaluat els següents models de classificació:

1. **Regressió Logística (One-vs-Rest)**
2. **SVM (Support Vector Machine)** amb kernel lineal
3. **KNN (k-Nearest Neighbours)** amb distància euclidiana
4. **Decision Tree (Arbre de decisió)**

Per a cada model:
- s’utilitza el mateix preprocessament
- s’avalua el rendiment mitjançant:
  - accuracy
  - matriu de confusió
  - `classification_report`

## Serialització dels models

Els models entrenats es serialitzen amb la llibreria **`joblib`**, juntament amb tots els elements necessaris per poder fer prediccions correctes posteriorment:

- model entrenat
- `LabelEncoder`
- `DictVectorizer`
- `StandardScaler`
- llista de variables numèriques

Els fitxers generats són:

models/
├── logistic_regression.joblib
├── svm.joblib
├── knn.joblib
└── decision_tree.joblib

## Servei web amb Flask

S’ha desenvolupat un servei web REST amb **Flask** que carrega els models serialitzats i permet fer prediccions sobre nous exemplars de pingüins mitjançant peticions HTTP **POST** en format JSON.

### Endpoints disponibles

| Model | Endpoint |
|------|----------|
| Regressió logística | `/predict/logistic_regression` |
| SVM | `/predict/svm` |
| KNN | `/predict/knn` |
| Arbre de decisió | `/predict/decision_tree` |

### Exemple de JSON d’entrada

```json
{
  "culmen_length_mm": 39.1,
  "culmen_depth_mm": 18.7,
  "flipper_length_mm": 181,
  "body_mass_g": 3750,
  "island": "Torgersen",
  "sex": "Male"
}
```

## Client Python

El projecte inclou un client en Python que consumeix el servei Flask:

realitza com a mínim dues peticions per cada model

mostra les respostes per consola

utilitza la llibreria requests

Això permet validar el correcte funcionament del servei web i dels models desplegats.

## Estructura del projecte
Tasca_3_Maite_Ladaria/
├── dataset/
│   └── penguins_size.csv
├── Images/
│   └── *.png
├── models/
│   ├── logistic_regression.joblib
│   ├── svm.joblib
│   ├── knn.joblib
│   └── decision_tree.joblib
├── notebook/
│   ├── data_utils.py
│   ├── Regresion/
│   ├── VectorMachine/
│   ├── KNN/
│   └── DecisionTree/
├── flask_app/
│   └── app.py
├── client/
│   └── client.py
├── environment.yml
└── README.md

## Com executar el projecte
1) Crear l’entorn Conda
conda env create -f environment.yml
conda activate Maite_Ladaria_penguin_task

2) Entrenar i serialitzar els models

Executa cada script dins la seva carpeta corresponent. Per exemple:

cd notebook/Regresion
python Regresion.py


Repeteix el procés per a VectorMachine, KNN i DecisionTree.

3) Arrencar el servidor Flask
cd flask_app
python app.py


El servidor quedarà disponible a:

http://127.0.0.1:5000

4) Executar el client
cd client
python client.py

Tecnologies utilitzades

Python

NumPy

Pandas

scikit-learn

Flask

joblib

requests

matplotlib

Autoria

Maite Ladaria
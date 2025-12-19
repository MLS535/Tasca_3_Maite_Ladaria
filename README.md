# 🐧 Clasificación de Pingüinos – Palmer Penguins

Proyecto de **Sistemas de Aprendizaje Automático** donde se implementan y despliegan distintos **modelos de clasificación supervisada** sobre el dataset *Palmer Penguins*, siguiendo como referencia el caso práctico del dataset Iris.

El proyecto incluye:

* Preprocesamiento completo de datos
* Entrenamiento de **4 modelos de clasificación**
* Serialización de modelos
* Despliegue de un **servicio web con Flask**
* Cliente Python que consume el servicio
* Gráficas en la parte de Images
---

## 📊 Dataset

Se utiliza el dataset **Palmer Penguins**, descargado desde Kaggle (`penguins_size.csv`).

Cada individuo contiene las siguientes variables:

* `species` → **variable objetivo** (Adelie, Chinstrap, Gentoo)
* `island` → Dream, Torgersen, Biscoe
* `culmen_length_mm`
* `culmen_depth_mm`
* `flipper_length_mm`
* `body_mass_g`
* `sex` → Male / Female

Las filas con valores `NA` se eliminan para simplificar el proceso.

---

## ⚙️ Preprocesamiento de datos

El preprocesamiento es **común para todos los modelos** y se realiza en el módulo `data_utils.py`:

* Eliminación de valores nulos
* División del dataset:

  * **80% entrenamiento**
  * **20% prueba**
* Codificación de la variable objetivo (`species`) mediante `LabelEncoder`
* Codificación *one-hot* de variables categóricas (`island`, `sex`) con `DictVectorizer`
* Normalización de variables numéricas con `StandardScaler`

  * Media = 0
  * Desviación típica = 1
  * Ajuste del escalado **solo con el conjunto de entrenamiento**

Este enfoque evita *data leakage* y garantiza coherencia entre entrenamiento y predicción.

---

## 🤖 Modelos implementados

Se han entrenado y evaluado los siguientes clasificadores:

1. **Regresión Logística (One-vs-Rest)**
2. **Máquinas de Soporte Vectorial (SVM)** – kernel lineal
3. **K-Nearest Neighbours (KNN)** – distancia euclídea
4. **Árboles de Decisión (Decision Tree)**

Cada modelo:

* Se entrena usando el mismo preprocesamiento
* Se evalúa con:

  * Accuracy
  * Matriz de confusión
  * Classification report

---

## 💾 Serialización de modelos

Los modelos entrenados se serializan utilizando **`joblib`**, junto con todos los elementos necesarios para la predicción:

* Modelo entrenado
* `LabelEncoder`
* `DictVectorizer`
* `StandardScaler`
* Lista de variables numéricas

Los modelos se guardan en la carpeta:

```
models/
├── logistic_regression.joblib
├── svm.joblib
├── knn.joblib
└── decision_tree.joblib
```

---

## 🌐 Servicio web – Flask

Se ha desarrollado un servicio web REST con **Flask** que carga los modelos serializados y permite realizar predicciones sobre nuevos pingüinos.

### Endpoints disponibles

Todos los endpoints aceptan peticiones **POST** con datos en formato JSON:

| Modelo              | Endpoint                       |
| ------------------- | ------------------------------ |
| Regresión logística | `/predict/logistic_regression` |
| SVM                 | `/predict/svm`                 |
| KNN                 | `/predict/knn`                 |
| Árbol de decisión   | `/predict/decision_tree`       |

### Ejemplo de petición JSON

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

---

## 🧑‍💻 Cliente Python

Se ha implementado un cliente en Python que interactúa con el servicio Flask:

* Realiza **al menos dos peticiones por modelo**
* Muestra las respuestas por consola
* Utiliza la librería `requests`

Esto permite validar el correcto funcionamiento del servicio web y los modelos desplegados.

---

## 📁 Estructura del proyecto

```
Tasca 3/
├── dataset/
│   └── penguins_size.csv
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
└── README.md
```

---

## ▶️ Ejecución del proyecto

### 1️⃣ Entrenar y serializar modelos

Ejecutar cada script de modelo desde su carpeta correspondiente.

### 2️⃣ Arrancar el servidor Flask

```bash
cd flask_app
python app.py
```

Servidor disponible en:

```
http://127.0.0.1:5000
```

### 3️⃣ Ejecutar el cliente

```bash
cd client
python client.py
```

---

## 🧪 Tecnologías utilizadas

* Python
* Pandas
* NumPy
* scikit-learn
* Flask
* joblib
* requests

---

## 📌 Autoría

Proyecto desarrollado como parte del módulo **Sistemes d’Aprenentatge Automàtic**, utilizando el dataset Palmer Penguins como alternativa moderna al clásico Iris.

---

## ✅ Estado del proyecto

✔️ Preprocesamiento correcto
✔️ 4 modelos entrenados
✔️ Modelos serializados
✔️ Servicio web desplegado
✔️ Cliente funcional

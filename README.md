# 🩺 Breast Cancer Detection API

A Deep Learning project for predicting whether a breast tumor is **Benign** or **Malignant** using the Breast Cancer Wisconsin Dataset. The project includes model training, preprocessing, evaluation, and deployment through a FastAPI REST API.

---

## 📌 Project Overview

This project demonstrates an end-to-end Deep Learning workflow:

* Data preprocessing
* Building a Neural Network with TensorFlow/Keras
* Model evaluation
* Building a REST API using FastAPI

---

## 📊 Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

The dataset contains **30 numerical features** extracted from digitized images of breast mass cell nuclei.

Target classes:

* **0 → Benign**
* **1 → Malignant**

---

## 🧠 Model

Framework:

* TensorFlow / Keras

Architecture:

* Sequential Neural Network
* Dense Layers
* ReLU Activation
* Sigmoid Output Layer

Loss Function:

* Binary Crossentropy

Optimizer:

* adam

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC Curve
* AUC

---

## 📈 Model Performance

Final Test Accuracy:

**99.12%**

Confusion Matrix:

```
[[72  0]
 [ 1 41]]
```

Classification Report:

```
Accuracy : 99.12%
Precision: 99%
Recall   : 99%
F1 Score : 99%
```

---

## 🚀 Running the API

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

FastAPI automatically generates interactive API documentation using Swagger UI.

---

## 🔍 Prediction Endpoint

**POST**

```
/predict
```

Example Request:

```json
{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  "perimeter_mean": 122.8,
  "area_mean": 1001.0,
  "smoothness_mean": 0.1184,
  "compactness_mean": 0.2776,
  "concavity_mean": 0.3001,
  "concave_points_mean": 0.1471,
  "symmetry_mean": 0.2419,
  "fractal_dimension_mean": 0.0787,

  "radius_se": 1.095,
  "texture_se": 0.9053,
  "perimeter_se": 8.589,
  "area_se": 153.4,
  "smoothness_se": 0.0064,
  "compactness_se": 0.049,
  "concavity_se": 0.0537,
  "concave_points_se": 0.0159,
  "symmetry_se": 0.030,
  "fractal_dimension_se": 0.0062,

  "radius_worst": 25.38,
  "texture_worst": 17.33,
  "perimeter_worst": 184.6,
  "area_worst": 2019.0,
  "smoothness_worst": 0.1622,
  "compactness_worst": 0.6656,
  "concavity_worst": 0.7119,
  "concave_points_worst": 0.2654,
  "symmetry_worst": 0.4601,
  "fractal_dimension_worst": 0.1189
}
```

Example Response:

```json
{
  "prediction": 1,
  "probability": 0.9987
}
```

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Uvicorn
* Git
* GitHub

---

## 📚 Learning Outcomes

Through this project, I practiced:

* Data preprocessing
* Feature scaling
* Neural Network development
* Model evaluation and interpretation
* Model persistence
* REST API development
* Git & GitHub workflow
* Preparing models for deployment

---

## 👤 Author

**Mohamed Ayman**

GitHub:
https://github.com/MooAyman

from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import os

# --------------------------
# 1. Cargar los datos
# --------------------------
df = load_data()
print("Primeras filas del dataset:")
print(df.head())

# --------------------------
# 2. Verificar y manejar valores nulos
# --------------------------
print("\nValores nulos antes del tratamiento:")
print(df.isnull().sum())
df.fillna(df.median(numeric_only=True), inplace=True)

# --------------------------
# 3. Separar variables predictoras y objetivo
# --------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

input_example = X[:5]  # ✅ Guardar muestra con nombres de columnas

# --------------------------
# 4. Escalado
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 5. División entrenamiento/prueba
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTamaño del conjunto de entrenamiento: {X_train.shape}")
print(f"Tamaño del conjunto de prueba: {X_test.shape}")

# --------------------------
# 6. Entrenamiento del modelo
# --------------------------
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# --------------------------
# 7. Predicción y evaluación
# --------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"✅ F1 Score: {f1:.4f}")

# --------------------------
# 8. Registro con MLflow
# --------------------------
import shutil

# Limpia la carpeta completamente si existe
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")

mlflow.set_tracking_uri("file:./mlruns")  # Reestablece tracking
mlflow.set_experiment("Default")  # Se asegura de usar la nueva ruta
 # Compatible con GitHub Actions

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    signature = infer_signature(X_test, y_pred)

    mlflow.sklearn.log_model(
        model,
        artifact_path="xgb_model",
        signature=signature,
        input_example=input_example
    )

    print("\n📦 Modelo registrado con MLflow.")


